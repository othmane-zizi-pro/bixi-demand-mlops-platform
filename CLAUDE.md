We are master's students at the Desautels Faculty of Management at McGill University.

Team (names + GitHub ids; personal contact details intentionally omitted from this public repo):

- Othmane Zizi — GitHub: othmane-zizi-pro
- Sarah Liu — GitHub: (to confirm)
- Ruihe Zhang (Louis) — GitHub: mudkipython
- Rui Zhao — GitHub: ruizhaoca

We must read the assignment instructions along with all the other files, i.e.:

- @instructions_final_project.md
- @instructions_final_project_presentation.md

The course materials are in:
- @llm-wiki-kafka

We must draw from them to ensure understanding and application of course concepts.

The master build plan lives in the main directory:
- @plan.md

The plan (v2) follows the team's "Methodology Improvement & Labor Division Plan" and includes:
- one clear owner per phase (Sarah + Louis on data; Othmane on predictive modeling + drift; Rui on
  clustering and the Streamlit/serving/deployment work),
- a general how-to on how to authenticate to GitHub from the coding agent,
- who handles what phase,
- each phase in a Pull Request on its own feature branch,
- a handoff/kickoff prompt so each teammate can just tell the agent their name and it picks up the right
  phase from @plan.md.

How to start your phase: tell your coding agent "My name is X. Let's ship the phase I'm responsible for,"
then let it read @plan.md and execute that phase. Using Codex instead of Claude Code? Copy this file's
contents into a local CODEX.md first so the agent gets the same context.

This is the GitHub repo: https://github.com/ruizhaoca/bixi-demand-mlops-platform

We push the following to the remote repo:

- llm wiki folder (@llm-wiki-kafka)
- @plan.md
- CLAUDE.md (this file, redacted)
- @instructions_final_project.md
- @instructions_final_project_presentation.md

The plan is built to score a 100% grade against the rubric in @instructions_final_project.md — while
deliberately scoping to a strong, high-value subset of topics rather than padding it with every reference
topic (e.g. causal inference and semi-supervised learning are intentionally out of scope).

All production deployment is handled with CDK or Terraform (infrastructure as code) for AWS. The AWS account
is accessed via IAM Identity Center (SSO). When everything is ready, deployment is run via the AWS CLI against
the CDK/Terraform code under `infra/` — see the Deployment Runbook (§8) in @plan.md.

Commits and Pull Requests must be authored by the human teammate driving the agent (their name + GitHub
account), never by the coding agent — so all of our names appear on the work.

If there is a report to produce, we produce it in LaTeX and convert it to PDF with tectonic.
