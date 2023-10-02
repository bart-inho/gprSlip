function [m,nb] = water_inclusion(l,h,LWC,s_max,hb,f)

    % l = model extension [lx ly lz] (in m)
    % h = model discretization (in m, same for all dimensions)
    % LWC is the liquid water content in %
    % s_max = maximum radius of inclusion (in m). remark: no minimal size
    % hb is the bedrock thickness (we don't want water inclusions where the bedrock is)
    % f is the fraction of water saturated layer, i.e. the ice layer that contains
    %the water inclusions (starting from the bedrock). Example: f=0.5 set the
    %water inclusions only on the lower half. f=0.3 display water inclusions on
    %one third of the ice thickness above bedrock
    
    %Note: we can actually set layering as we deal with bedrock
    
    %outputs:
    % m a logical array with 1 for water presence
    % lwc is the liquid water content in % (for 2D or 3D)
    
    %Model geometry
    
    lx= l(2); %length in meters
    ly= l(1); %height of total model
    X = lx/h; %image size along x axis, in pixel
    Y = ly/h; %image size along y axis, in pixel
    
    % Scatter inclusions
    
    [col, rows] = meshgrid(1:X, 1:Y);
    
    % create random location and size of inclusion
    m=false(Y,X); % m is a 2D "logical" matrix of the "dry" water layer (without inclusion).
    
    liq = 0; %initation of liquid water content
    nb=0; %initial counts of scatterers
    while liq < LWC 
        x = rand*X; %center location in pixel
        y = rand*Y; %note that repeatability is allowed, but shouldn'be an issue for small LWC values as it is in our case (~1%)
        r = rand*s_max; %in meters
        if y < Y - hb/h && y > Y*(1-f) % we only create water inclusion above bedrock and below the water saturated layer defined by f
          m = or(m,((rows - y).^2 + (col - x).^2 <= (r/h).^2)); %in pixel
          liq = liq + (pi*r^2/(l(1)*l(2)))*100;   %(pi*r^2/(l(1)*l(2)))*100 is the LWC in 2D in % (m2)
          nb=nb+1;
        end
        
    end
    
    nb=nb-1;%nb was incremented by one before exiting the while loop
    end
    
    
    
    