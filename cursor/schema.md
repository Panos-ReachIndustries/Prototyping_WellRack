# Subagent Schema

This document defines the subagent architecture, shared rules, and artifacts.
It is intended to be a practical schema for autonomous operation.

## Schema overview
- All agents inherit global rules from cursor/agent_instructions.md.
- Role-specific rules live in cursor/subagents/<role>.md.
- Shared templates live in cursor/tasks/.

## Subagent definition
Required sections for each role file:
- Mission
- Scope (allowed and prohibited actions)
- Inputs
- Outputs
- Process
- Approvals
- Handoff

YAML example:
```yaml
subagent:
  id: planner
  name: Planner
  mission: Decompose tasks and map dependencies.
  scope:
    allowed:
      - read repository files
      - identify risks and assumptions
    prohibited:
      - modify code
      - run destructive commands
  tools:
    read: [ReadFile, rg, Glob, LS]
    write: []
    run: []
  approvals_required:
    - change agent rules or safety constraints
  inputs:
    - user_request
    - repository_context
  outputs:
    - task_plan
    - assumptions
    - risks
  handoff:
    to_roles: [implementer]
    format: cursor/tasks/handoff_template.md
```

## Task definition
Tasks should be described using the intake template.

YAML example:
```yaml
task:
  id: T-001
  title: "Add subagent architecture docs"
  requester: user
  priority: medium
  constraints:
    - no internet access
    - no destructive operations
  acceptance_criteria:
    - schema documented in cursor/schema.md
    - subagent roles documented in cursor/subagents/
  status: planned
  assigned_roles:
    - planner
    - implementer
    - reviewer
```

## Handoff definition
Handoffs must be short, explicit, and action-ready.

YAML example:
```yaml
handoff:
  from_role: planner
  to_role: implementer
  summary: "Implement schema and role docs."
  files_touched: []
  assumptions:
    - no new dependencies
  risks:
    - scope creep
  approvals_needed:
    - none
```

## Review definition
Reviews must prioritize correctness and safety.

YAML example:
```yaml
review:
  scope: "cursor/ folder"
  findings:
    - severity: high
      file: cursor/schema.md
      line: 42
      issue: "Missing approval guard."
  residual_risks:
    - "No tests run"
```

## Runbook definition
Runner executions must be logged with command and output.

YAML example:
```yaml
runbook:
  command: "pytest -q"
  purpose: "Run test suite"
  expected_duration: "short"
  destructive: false
  approvals_required: false
  output_summary: "All tests passed"
```

## Conventions
- Use ASCII-only text in all artifacts.
- Keep role boundaries strict: no cross-role actions.
- Prefer minimal, reversible changes.
- Escalate to human approval for any action outside scope.
