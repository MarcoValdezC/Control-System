function currentPath=GetCurrentPath()
scriptName = mfilename('fullpath');
currentPath= fileparts(scriptName);
end
