You are an observer watching an expert complete a ServiceNow work task in a browser, trying to understand the reasoning behind each action so you could later explain the work to a trainee. You see the task ticket, the expert's recent actions, the page region around the element acted on, and the action the expert JUST took.

Infer why the expert took this action, and be honest about how sure you are. Return JSON only:
{
  "rationale": "your single best explanation of why the expert did this (one sentence, grounded in the ticket / protocol / page)",
  "confidence": 0.0-1.0,            // how confident you are that this rationale is the actual reason
  "alternative_rationales": ["..."], // OTHER reasons that are also genuinely consistent with what you can see; [] if the reason is clear
  "subgoal": "short label for the sub-goal this action serves, e.g. 'find the protocol article', 'delete the later duplicate'",
  "is_new_subgoal": true|false,      // does this action start a new sub-goal compared with the recent actions?
  "criticality": 0.0-1.0            // how consequential / hard to undo this action is (deleting or saving a record ≈ 1, navigating ≈ 0.2)
}
