# Delivery checklist

- [x] All 125 data pairs audited and frozen split created.
- [x] Three training-subject preprocessing overlays visually reviewed.
- [x] GPU 96³ forward/backward smoke check passes (four patches, RTX 5070).
- [ ] Complete training run; best validation checkpoint and logs retained.
- [x] Both BET variants produce 25 native-grid binary masks each.
- [ ] One frozen held-out analysis produces 75 metric rows and three requested figures.
- [ ] README and summary include measured values and exact consistent resume sentence.
- [ ] Medical resume placeholders replaced with measured results and one-page compilation verified.
- [ ] Public GitHub repository `monai-brain-extraction` created and pinned to the profile.
- [ ] Weights and prediction masks attached to a release with SHA-256 hashes.

Publication is an external action; the local repository and release files should be reviewed before publishing. No medical resume or personal contact information belongs in the project repository.

After GitHub CLI login and committing the final reviewed results, run `powershell -ExecutionPolicy Bypass -File scripts/publish.ps1` from the project directory. The helper checks the acceptance report, refuses a conflicting existing repository/remote, pushes the project, and creates a release with weights, masks and checksums. Pin the repository with GitHub profile → **Customize your pins**. If your OAuth token cannot push the optional GitHub Actions workflow, authenticate with the workflow scope or publish without that optional workflow.
