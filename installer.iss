; AirPoint Windows installer (Inno Setup).
;
; Wraps the PyInstaller onedir bundle (dist\AirPoint) into a single
; AirPoint-Setup.exe that installs per-user (no admin), creates Start Menu /
; Desktop shortcuts, and crucially keeps AirPoint.exe TOGETHER with its
; _internal folder in a dedicated install directory. That makes the
; "Failed to load Python DLL" error (exe separated from _internal) impossible,
; and gives the auto-updater its own folder to swap safely.
;
; Version is injected by CI:  ISCC /DMyAppVersion=1.0.4 installer.iss

#define MyAppName "AirPoint"
#define MyAppExeName "AirPoint.exe"
#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif

[Setup]
; Stable identifier so upgrades/uninstall recognise the same app across versions.
AppId=AirPoint.CoderKavin
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher=Kavin Venkat
; Per-user install location — no admin needed, and the auto-updater can write here.
DefaultDirName={localappdata}\Programs\AirPoint
PrivilegesRequired=lowest
DisableProgramGroupPage=yes
DisableDirPage=yes
OutputDir=installer_out
OutputBaseFilename=AirPoint-Setup
SetupIconFile=assets\icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
; Offer to close a running AirPoint when re-installing/upgrading in place.
CloseApplications=yes

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional shortcuts:"

[Files]
Source: "dist\AirPoint\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{autoprograms}\AirPoint"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\AirPoint"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Launch AirPoint"; Flags: nowait postinstall skipifsilent
