# Stage 6 test repository (steps for the project owner)

Goal: a public GitHub repository `BilalAamir4/mapr-playground` with one open pull request, so the
stage 6 acceptance run (decision 0016) can post a real review. The project is the synthetic
`tests/fixtures/dev_risky_pr` fixture, which matches the `dev_fixture` sandbox image (pass
`--sandbox-repo dev_fixture`). Nothing here touches the evaluation.

The fixture's `workspace/` is the project **after** the PR, so the base version is made by
reversing `pr.diff` in a copy outside this repository.

## 1. Create the repository

On GitHub: **New repository** → owner `BilalAamir4`, name `mapr-playground`, **Public**, and
**no** README, `.gitignore` or license (it must be empty).

## 2. Push the base version to `main`

Run in PowerShell. The copy lives in your home folder, not in the reviewer repository.

```powershell
$fixture = "C:\Users\Bilal Aamir\Documents\GitHub\Multi-Agent-PR-Reviewer\tests\fixtures\dev_risky_pr"
Copy-Item "$fixture\workspace" "$HOME\mapr-playground" -Recurse
Copy-Item "$fixture\pr.diff" "$HOME\risky-pr.diff"
Set-Location "$HOME\mapr-playground"
git init -b main
git config core.autocrlf false
```

Turn the copy into the base version (reverse the PR), then commit and push it:

```powershell
git apply -R "$HOME\risky-pr.diff"
git add .
git commit -m "Base version"
git remote add origin https://github.com/BilalAamir4/mapr-playground.git
git push -u origin main
```

## 3. Push the PR branch and open the pull request

```powershell
git switch -c risky-change
git apply "$HOME\risky-pr.diff"
git add .
git commit -m "Add per-account entries, running balances and a nightly sync URL"
git push -u origin risky-change
```

On GitHub open the **Compare & pull request** banner for `risky-change` → base `main`:

- Title: `Add per-account entries, running balances and a nightly sync URL`
- Description: `Adds entries_for(), running_balances() and sync_url(), and updates the report tests.`

Create the pull request and note its number (the last part of the URL, `.../pull/N`).

## 4. Create the token

GitHub → Settings → Developer settings → Personal access tokens → **Fine-grained tokens** →
**Generate new token**:

- Name: `mapr-playground-reviewer`; **Expiration: 30 days**.
- Resource owner: `BilalAamir4`. **Repository access: Only select repositories** →
  `mapr-playground` only.
- Repository permissions: **Pull requests: Read and write**, **Contents: Read-only**,
  **Metadata: Read-only** (selected automatically). Nothing else.

Copy the token once and put it in `.env` in the reviewer repository (`.env` is gitignored):

```
GITHUB_TOKEN=<the token>
```

Do not paste the token into the chat, an issue or a log.

## 5. Reply

Tell me the pull request number. I then run the acceptance run (Task 6).
