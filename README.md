TRY TO GET VISUAL STUDIO BUILD TOOLS for VISUAL STUDIO 2017-2022 (2022) preferred for cu128: https://aka.ms/vs/17/release/vs_buildtools.exe


Just a quick setup guide layout: 
<img width="1344" height="837" alt="image" src="https://github.com/user-attachments/assets/721c906c-dc7a-4c9b-b74f-c394ec702e77" />








🛠️ Troubleshooting & Requirements
🐍 If Python is missing:
If you see "Python was not found", run this command, then restart PowerShell:

winget install Python.Python.3.11 --override "/quiet InstallAllUsers=1 PrependPath=1"
🟢 If Node.js (npm) is missing:
If npm install fails, run this command, then restart PowerShell:

winget install OpenJS.NodeJS
🏗️ If "Something went wrong" (Bundled Python):
If the app can't find its internal Python files, run this helper script:

cd "$HOME\modly"
node scripts/download-python-embed.js
###NOTE IF EXIT CODE -1 FROM CUDA EVNIRONMENT, ADD "CUDA_HOME" to your system variables and path it to your CUDA file, i.e. "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x"

🔌 Extensions (Hunyuan3D)
Installation Steps:
Verify Git: Ensure C:\Program Files\Git\cmd is in your System Path.
Permissions: Do not install Modly in a OneDrive folder. This causes permission errors.
Download Weights: Open the Modly extensions panel and click the Purple Download Button.
Crucial: Stay on the tab until finished. Restart Modly after the download completes.
Missing Models: If components are present but "not found," install the VC Redistributable.
💻 Hardware & Performance
VRAM: 6GB (Minimum) | 8GB+ (Recommended).
Efficiency: The Turbo model is more memory-efficient than Standard.
Updates: Currently, multi-image input is being patched. Until then, the system defaults to a single front-view image.


-If you get a compilation error while building the C++ extension custom_rasterizer, please contact me on discord via the community extensions tab as this is likely due to running the app instead of going through the launch.bat directly. I can fix that if anyone has issues with that. 




