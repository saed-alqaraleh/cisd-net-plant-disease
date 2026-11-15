# Upload the vetted Git bundle to the private GitHub repository

The Git bundle is a single self-contained snapshot of the vetted CISD-Net
source tree and numerical evidence. It contains one initial commit on `main`
and no history from any other research repository.

The destination GitHub repository must be `saed-alqaraleh/cisd-net-plant-disease`,
with visibility **Private**. Do not run these commands in a different project.

## Windows PowerShell — safe isolated upload

Download `CISD_Net_Private_Ready.git.bundle` to your Downloads folder, then
open **PowerShell** and run:

```powershell
$Bundle = Join-Path $HOME 'Downloads\CISD_Net_Private_Ready.git.bundle'
$Target = Join-Path $HOME 'Downloads\cisd-net-private-upload'
Test-Path $Bundle
Test-Path $Target
```

The first result must be `True`; the second must be `False`. If the second
result is `True`, use a different new folder name rather than deleting or
overwriting anything.

```powershell
git clone -b main $Bundle $Target
git -C $Target remote set-url origin https://github.com/saed-alqaraleh/cisd-net-plant-disease.git
git -C $Target remote -v
git -C $Target status --short
git -C $Target log -1 --oneline
git -C $Target push --dry-run origin main
```

Check that **both** fetch and push URLs refer to exactly
`saed-alqaraleh/cisd-net-plant-disease.git` and the status is clean. The dry
run must not show rejection or any forced update. Only then run:

```powershell
git -C $Target push -u origin main
```

Never run `--force`, `--mirror`, `--all`, `git clean`, or `git reset --hard`
for this upload. This process does not call `git add .` anywhere and does not
touch other local working directories or GitHub repositories.

## Verify after upload

Clone the private GitHub repository into a **different new folder** and run:

```powershell
python -m pip install -r requirements-verify.txt
python scripts/verify_reproduction.py
python -m unittest discover -s tests -v
```

Confirm 99/99 conformal set replay, `SHA256SUMS: OK`, and `VERIFICATION PASSED`.
Keep the repo Private until the science, rights, citation and metadata checks
in `docs/RELEASE_CHECKLIST.md` are complete. Do not publish a GitHub Release,
attach the 75.6 MB cache, or register a Zenodo DOI before approval.
