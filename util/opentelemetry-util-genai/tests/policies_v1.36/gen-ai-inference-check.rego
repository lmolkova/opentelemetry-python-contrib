package live_check_advice

import rego.v1

inference_operations := {"chat", "generate_content", "text_completion"}
expected_attributes := {
  "gen_ai.operation.name",
  "gen_ai.request.model",
  "gen_ai.system",
  "gen_ai.response.model",
  "gen_ai.response.id",
  "gen_ai.response.finish_reasons",
  "gen_ai.usage.input_tokens",
  "gen_ai.usage.output_tokens",
  "server.address",
}

deny contains make_finding("unexpected_gen_ai_operation", context, message) if {
  input.sample.span
  operation_name = get_attribute_value(input.sample.span.attributes, "gen_ai.operation.name")
  not operation_name in inference_operations

  message := sprintf("Attribute 'gen_ai.operation.name' has an unexpected value - %v.", [operation_name])
  context := {"attribute_name": "gen_ai.operation.name", "attribute_value": operation_name}
}

deny contains make_finding("attribute_not_found", context, message) if {
  input.sample.span
  expected_attribute := expected_attributes[_]
  get_attribute_value(input.sample.span.attributes, expected_attribute) == null
  message := sprintf("Attribute '%v' not found.", [expected_attribute])
  context := {"attribute_name": expected_attribute}
}

deny contains make_finding("wrong_span_name", context, message) if {
  input.sample.span
  name = input.sample.span.name
  operation_name := get_attribute_value_or(input.sample.span.attributes, "gen_ai.operation.name", "")
  model_name := get_attribute_value_or(input.sample.span.attributes, "gen_ai.request.model", "")
  expected_name := concat(" ", [operation_name, model_name])

  not name == expected_name
  message := sprintf("Span name '%v' does not match expected name '%v'.", [name, expected_name])
  context := {"expected_name": expected_name, "actual_name": name}
}

make_finding(id, context, message) := {
  "type": "advice",
  "advice_type" : id,
  "advice_level": "violation",
  "context": context,
  "message": message
}

get_attribute_value(attributes, name) := value if {
    attr := attributes[_]
    attr.name == name
    value := attr["value"]
} else = null

get_attribute_value_or(attributes, name, def) := value if {
    attr := attributes[_]
    attr.name == name
    value := attr["value"]
} else = def