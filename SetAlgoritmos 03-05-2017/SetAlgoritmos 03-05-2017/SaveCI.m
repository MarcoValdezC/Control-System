function SaveCI(time,RUNS,G_MAX,SAVE_G,NP,D,D_MIN,D_MAX,run)
global ALG_NAME;
global P_DIR;
Parametros;
%Save workspace
save(sprintf('resultados/%s_%s_Run%d_CI.mat',ALG_NAME,P_DIR,run),'time','Cond_Algoritmo','ALG_NAME','G_MAX','SAVE_G','NP','D','D_MIN','D_MAX');
end


