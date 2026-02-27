# Cursor Subagent Architecture

This document defines how autonomous subagents operate in this repository.
It is the source of truth for rules, roles, and coordination.

## Purpose and scope
This repository is a private AI/CV/Agentic R&D workspace and lab notebook.
It is not a polished product repo.

## Guardrails and boundaries (global rules)
- No internet access unless explicitly stated in a task.
- No access to secrets, credentials, or private tokens.
- No destructive operations (deleting repos, force-push, infra changes).
- No data exfiltration or copying company IP outside the repo.
- Keep product-specific implementations in company repos or clearly separated
  folders.

## Autonomy level and approvals
Default mode is semi-autonomous:
- Agents may plan and implement independently.
- Human approval is required before:
  - Merging PRs
  - Running destructive or irreversible actions
  - Changing agent rules or safety constraints

## Core roles and responsibilities
Each role is defined in cursor/subagents/.
- Planner: task decomposition, dependency analysis, risk mapping
- Implementer: code changes only
- Reviewer: correctness, safety, and regression review
- Runner: execution, tests, CI, commands
- Data-Analyst: data inspection, metrics, debugging support

## Coordination workflow (default)
1. Intake: capture goals, constraints, and acceptance criteria.
2. Plan: decompose work and identify dependencies.
3. Implement: make code changes with minimal scope.
4. Review: verify correctness and safety.
5. Run: execute tests or commands as needed.
6. Report: summarize changes, risks, and test results.

Use templates in cursor/tasks/ for intake, handoffs, and reviews.

## Repository privacy and access notes
- Private repos are not visible to the public.
- Org membership does not automatically grant repo access.
- Access is controlled at the repo or team level.
- Org owners can see metadata but cannot read private contents unless granted
  access.

## Directory layout
```
cursor/
├── agent_instructions.md
├── schema.md
├── subagents/
│   ├── planner.md
│   ├── implementer.md
│   ├── reviewer.md
│   ├── runner.md
│   └── data_analyst.md
└── tasks/
    ├── task_intake.md
    ├── plan_template.md
    ├── handoff_template.md
    ├── review_checklist.md
    └── runbook_template.md
```
