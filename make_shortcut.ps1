# Creates the "Time Cube" desktop shortcut (pythonw + launch.pyw, custom icon).
$app = Split-Path -Parent $MyInvocation.MyCommand.Path
$pyw = (Get-Command pythonw -ErrorAction SilentlyContinue).Source
if (-not $pyw) { $pyw = Join-Path (Split-Path -Parent (Get-Command python).Source) 'pythonw.exe' }
$desk = [Environment]::GetFolderPath('Desktop')
$sh = New-Object -ComObject WScript.Shell
$lnk = $sh.CreateShortcut((Join-Path $desk 'Time Cube.lnk'))
$lnk.TargetPath = $pyw
$lnk.Arguments = '"' + (Join-Path $app 'launch.pyw') + '"'
$lnk.WorkingDirectory = $app
$lnk.IconLocation = (Join-Path $app 'icon.ico') + ',0'
$lnk.Description = 'Time Cube - video frames stacked into a tumbling 3D cube (wire, slices, solid), MP4 and PNG export'
$lnk.WindowStyle = 1
$lnk.Save()
Write-Output "Shortcut created: $(Join-Path $desk 'Time Cube.lnk')"
