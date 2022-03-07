function Save(pop,run,g,SAVE_G)
global ALG_NAME;
global P_DIR;
    
if(mod(g,SAVE_G)==0)
    subpop=pop(:,:,g-SAVE_G+1:g);
    %Save workspace
    save(sprintf('resultados/%s_%s_Run%d_Gen%d_%d.mat',ALG_NAME,P_DIR,run,g-SAVE_G+1,g),'subpop');
end
end


