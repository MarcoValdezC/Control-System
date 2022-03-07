function SaveAll(pop,run,gi,gf)
global ALG_NAME;
global P_DIR;
%Save workspace
save(sprintf('resultados/%s_%s_Run%d_Gen%d_%d.mat',ALG_NAME,P_DIR,run,gi,gf),'pop');
end


