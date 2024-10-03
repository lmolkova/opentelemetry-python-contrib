# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from typing import Optional, Union
from opentelemetry.trace import SpanKind, Span
from opentelemetry._events import EventLogger, Event
from opentelemetry.trace.status import Status, StatusCode
from openai import NOT_GIVEN
from opentelemetry.semconv._incubating.attributes import gen_ai_attributes, error_attributes
from opentelemetry.semconv.attributes import error_attributes

from opentelemetry.trace import Tracer


def chat_completions_create(original_method, version, tracer: Tracer, event_logger: EventLogger):
    """Wrap the `create` method of the `ChatCompletion` class to trace it."""

    def traced_method(wrapped, instance, args, kwargs):
        operation_name = "chat"
        request_model = kwargs.get("model")

        span_name = f"{operation_name} {request_model}"

        with tracer.start_as_current_span(name=span_name, kind=SpanKind.CLIENT, end_on_exit=False) as span:
            if span.is_recording():
                span.set_attribute(gen_ai_attributes.GEN_AI_SYSTEM, "openai")
                span.set_attribute(gen_ai_attributes.GEN_AI_OPERATION_NAME, operation_name)
                span.set_attribute(gen_ai_attributes.GEN_AI_REQUEST_MODEL, request_model)
                _set_span_attribute(span, gen_ai_attributes.GEN_AI_REQUEST_TEMPERATURE, kwargs.get("temperature"))
                _set_span_attribute(span, gen_ai_attributes.GEN_AI_REQUEST_TOP_P, kwargs.get("top_p"))
                _set_span_attribute(span, gen_ai_attributes.GEN_AI_REQUEST_MAX_TOKENS, kwargs.get("max_tokens"))
                _set_span_attribute(span, gen_ai_attributes.GEN_AI_REQUEST_PRESENCE_PENALTY, kwargs.get("presence_penalty"))
                _set_span_attribute(span, gen_ai_attributes.GEN_AI_REQUEST_FREQUENCY_PENALTY, kwargs.get("frequency_penalty"))

                for msg in kwargs.get("messages", []):
                    event_logger.emit(_message_to_event(msg))

            # TODO: server.address, port

            try:
                result = wrapped(*args, **kwargs)
                if not span.is_recording():
                    return result

                if _is_streaming(kwargs):
                    return StreamWrapper(
                        result,
                        kwargs.get("n") or 1,
                        span,
                        event_logger
                    )
                else:
                    _set_response_attributes(span, result.model, result.id,
                                            result.usage.prompt_tokens if result.usage else None,
                                            result.usage.completion_tokens if result.usage else None)

                    finish_reasons = []
                    if hasattr(result, "choices") and result.choices is not None:
                        for choice in result.choices:
                            finish_reasons.append(choice.finish_reason or "error")

                            choice_content = JsonBody({
                                "index": choice.index,
                                "finish_reason": choice.finish_reason,
                                "message": _choice_to_message(choice),
                            })
                            event_logger.emit(Event(name="gen_ai.choice", body=choice_content))
                    span.set_attribute(gen_ai_attributes.GEN_AI_RESPONSE_FINISH_REASONS, finish_reasons)
                    span.end()
                    return result

            except Exception as error:
                _record_error(error, span)
                span.end()
                raise

    return traced_method

def _record_error(error, span):
    span.set_attribute(error_attributes.ERROR_TYPE, type(error).__qualname__)
    span.set_status(Status(StatusCode.ERROR, str(error)))
    span.end()
    raise error

def _choice_to_message(choice):
    message = {}

    if not hasattr(choice, "message"):
        return message

    if hasattr(choice.message, "content") and choice.message.content is not None:
        message["content"] = choice.message.content

    if hasattr(choice.message, "tool_calls") and choice.message.tool_calls is not None:
        message["tool_calls"] = [
            {
                "id": tool_call.id,
                "type": tool_call.type,
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                },
            }
            for tool_call in choice.message.tool_calls
        ]
    return message


def _get_prop(msg, prop):
    if isinstance(msg, dict):
        return msg.get(prop)
    else:
        return getattr(msg, prop, None)


def _message_to_event(message):
    role = _get_prop(message, "role")
    content = _get_prop(message, "content")
    if role == "user":
        return Event(name="gen_ai.user.message", body=JsonBody({"content": content}))
    elif role == "system":
        return Event(name="gen_ai.system.message", body=JsonBody({"content": content}))
    elif role == "assistant":
        tool_calls = _get_prop(message, "tool_calls")
        if not tool_calls:
            return Event(name="gen_ai.assistant.message", body=JsonBody({"content": content}))
        else:
            body = {"tool_calls": [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments,
                    },
                }
                for tool_call in tool_calls
            ]}
            if content:
                body["content"] = content
            return Event(name="gen_ai.assistant.message", body=JsonBody(body))
    elif role == "tool":
        return Event(name="gen_ai.tool.message", body=JsonBody({"content": content, "id": _get_prop(message, "tool_call_id")}))

def _set_response_attributes(span, response_model, response_id, usage_input_tokens, usage_output_tokens):
    if not span.is_recording():
        return

    _set_span_attribute(span, gen_ai_attributes.GEN_AI_RESPONSE_MODEL, response_model)
    _set_span_attribute(span, gen_ai_attributes.GEN_AI_RESPONSE_ID, response_id)
    _set_span_attribute(span, gen_ai_attributes.GEN_AI_USAGE_INPUT_TOKENS, usage_input_tokens)
    _set_span_attribute(span, gen_ai_attributes.GEN_AI_USAGE_OUTPUT_TOKENS, usage_output_tokens)

def _set_span_attribute(span: Span, name, value):
    if value is not None and (value != "" or value != NOT_GIVEN):
        span.set_attribute(name, value)

def _is_streaming(kwargs):
    stream = kwargs.get("stream")
    return bool(stream) and stream != NOT_GIVEN

class ChoiceBuffer:
    def __init__(self, index):
        self.index = index
        self.finish_reason = None
        self.content_str = ""
        self.role = None

    def append(self, choice):
        if (choice.finish_reason is not None):
            self.finish_reason = choice.finish_reason

        if choice.delta and choice.delta.content is not None:
            self.content_str += str(choice.delta.content)

        # TODOs
        #  - tool calls
        #  - multi-modal messages

class StreamWrapper:
    span: Span

    def __init__(
        self,
        stream,
        num_choices,
        span,
        event_logger: EventLogger,
    ):
        self.stream = stream
        self.span = span

        self.choices = [None] * num_choices
        for i in range(num_choices):
            self.choices[i] = ChoiceBuffer(i)
        self.completion_tokens = 0
        self.response_model = None
        self.response_id = None
        self.num_choices = num_choices
        self.event_logger = event_logger
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self._span_started = False
        self.setup()

    def setup(self):
        if not self._span_started:
            self._span_started = True

    def cleanup(self, exc_val=None):
        if self._span_started:
            _set_response_attributes(self.span, self.response_model, self.response_id, self.prompt_tokens, self.completion_tokens)

            finish_reasons = []

            for choice in self.choices:
                finish_reasons.append(choice.finish_reason or "error")
                choice_content = JsonBody({
                    "index": choice.index,
                    "finish_reason": choice.finish_reason,
                    "message": {"content" : choice.content_str}, # todo - do we need to deserialize?
                })
                self.event_logger.emit(Event(name="gen_ai.choice", body=choice_content, trace_id=self.span.get_span_context().trace_id, span_id=self.span.get_span_context().span_id))

            _set_span_attribute(self.span, gen_ai_attributes.GEN_AI_RESPONSE_FINISH_REASONS, finish_reasons)
            if (exc_val is not None):
                _record_error(exc_val, self.span)

            self.span.end()
            self._span_started = False

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup(exc_val)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            chunk = next(self.stream)
            self.process_chunk(chunk)
            return chunk
        except StopIteration:
            self.cleanup()
            raise
        except Exception as e:
            self.cleanup(e)
            raise

    def process_chunk(self, chunk):
        if (self.response_id is None):
            self.response_id = chunk.id

        if (self.response_model is None):
            self.response_model = _get_prop(chunk, "model")

        if getattr(chunk, "usage"):
            self.completion_tokens = chunk.usage.completion_tokens
            self.prompt_tokens = chunk.usage.prompt_tokens

        if hasattr(chunk, "choices") and chunk.choices is not None:
            for choice in chunk.choices:
                if choice:
                    self.choices[choice.index].append(choice)

class JsonBody(dict):
    def __init__(self, obj=None, **kwargs):
        if obj is None:
            obj = {}
        super().__init__(obj, **kwargs)

    def to_json(self):
        return json.dumps(self)

    def __str__(self):
        return self.to_json()