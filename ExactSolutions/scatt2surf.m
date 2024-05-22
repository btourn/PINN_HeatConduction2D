function [x1,y1,z_surf]=scatt2surf(xx,yy,zz)   
    x1= unique(round(xx,7));
    y1=unique(round(yy,7));
    z_surf=ones(length(x1),length(y1));
    for i=1:length(x1)
        for j=1:length(y1) 
        indx=[xx==x1(i) & yy==y1(j)];
        z_surf(i,j)=zz(indx);   
        end
    end
    z_surf=z_surf';
end