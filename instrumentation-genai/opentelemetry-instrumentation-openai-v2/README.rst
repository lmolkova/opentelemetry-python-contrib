OpenTelemetry OpenAI Instrumentation
====================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/opentelemetry-instrumentation-openai-v2.svg
   :target: https://pypi.org/project/opentelemetry-instrumentation-openai-v2/

Instrumentation with OpenAI that supports the OpenAI library and is
specified to trace_integration using 'OpenAI'.


Installation
------------

::

    pip install opentelemetry-instrumentation-openai-v2

Usage
-----

Instrumenting all clients
*************************

When using the instrumentor, all clients will automatically trace OpenAI chat completion operations
and capture prompts and completions as events.

Make sure to configure OpenTelemetry tracing, logging, and events to capture all telemetry emitted by the instrumentation.

.. code-block:: python

    from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor

    OpenAIInstrumentor().instrument()

    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Write a short poem on open telemetry."},
        ],
    )

Enabling message content
*************************

Message content such as the contents of the prompt, completion, function arguments and return values
are not captured by default. To enable message content, set the environment variable
`OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` to `true`.

Uninstrument
************

To uninstrument clients, call the uninstrument method:

.. code-block:: python

    from opentelemetry.instrumentation.openai_v2 import OpenAIInstrumentor

    OpenAIInstrumentor().instrument()
    # ...

    # Uninstrument all clients
    OpenAIInstrumentor().uninstrument()


References
----------
* `OpenTelemetry OpenAI Instrumentation <https://opentelemetry-python-contrib.readthedocs.io/en/latest/instrumentation/openai/openai.html>`_
* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `OpenTelemetry Python Examples <https://github.com/open-telemetry/opentelemetry-python/tree/main/docs/examples>`_

