# Delivery checklist

- [x] All 125 data pairs audited and frozen split created.
- [x] Three training-subject preprocessing overlays visually reviewed.
- [x] GPU 96³ forward/backward smoke check passes (four patches, RTX 5070).
- [x] Complete 200-epoch training run; epoch 190 selected by validation Dice (0.986003); checkpoint and logs retained.
- [x] Both BET variants produce 25 native-grid binary masks each.
- [x] One frozen held-out analysis produces 75 metric rows and three requested figures; supplementary BET failure review retained.
- [x] README and summary include measured values and exact consistent resume sentence, with the uncropped default/-R comparison qualification.
- [x] Fresh environment replay: 47 locked dependencies verified, eight tests pass, 75-row metrics CSV SHA-256 identical.
- [x] Medical resume placeholders replaced in `artifacts/resume_medical.tex`; original resume preserved outside the repository.
- [ ] One-page medical resume PDF verified. The built-in compiler returns `Unable to find standard directories for platform`; source is preserved and queued in the editor, but compilation/page count remain unverified.
- [x] [Public GitHub repository](https://github.com/hennyi-yin/monai-brain-extraction) created and pinned to the [profile](https://github.com/hennyi-yin).
- [x] Local release assets prepared: full best checkpoint, model-only weights, 75 masks and manifests, SHA-256 hashes.
- [x] Weights and prediction masks uploaded to [Release v1.0.0](https://github.com/hennyi-yin/monai-brain-extraction/releases/tag/v1.0.0). GitHub's asset SHA-256 digests match the local manifest.

The experiment, public repository, release upload and profile pinning are complete. One-page resume PDF verification remains blocked by the built-in compiler's platform-directory error. The medical resume stays in the ignored local artifact folder.

After GitHub CLI login and committing the final reviewed results, run `powershell -ExecutionPolicy Bypass -File scripts/publish.ps1` from the project directory. The helper checks the acceptance report, refuses a conflicting existing repository/remote, pushes the project, and creates a release with weights, masks and checksums. Pin the repository with GitHub profile → **Customize your pins**. If your OAuth token cannot push the optional GitHub Actions workflow, authenticate with the workflow scope or publish without that optional workflow.
