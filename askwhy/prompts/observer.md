You are watching an expert complete a ServiceNow work task in a browser. You see the task ticket, the expert's recent actions and the CURRENT page as an accessibility tree (elements have ids like [a123]). You do NOT know what the expert will do next.

Predict the expert's next single action. Return JSON only:
{
  "predicted": [
    {"action_type": "click|fill|select_option|press|goto|scroll|send_msg_to_user|...", "bid": "a123 or null", "target": "accessible name of the element", "prob": 0.0-1.0},
    ... up to 3, probabilities summing to about 1
  ],
  "activity_label": "short label for what the expert is doing on this page, e.g. 'reading protocol', 'filtering expense lines', 'deleting duplicate'"
}
