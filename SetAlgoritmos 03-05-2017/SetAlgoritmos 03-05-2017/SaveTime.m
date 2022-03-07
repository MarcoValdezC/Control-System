function SaveTime(time,run)
global ALG_NAME;
global P_DIR;
%Save workspace
save(sprintf('resultados/%s_%s_Run%d_Time.mat',ALG_NAME,P_DIR,run),'time');
end


