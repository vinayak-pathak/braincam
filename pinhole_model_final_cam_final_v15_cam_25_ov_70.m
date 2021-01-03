%% Simulating system by predetermining the number of cameras and overlap 9 cameras 70% Overlap


clear;
%Comment out the below line to rerun everything
%-load('C:\Users\COLDD03\OneDrive - Duke University\Codes\depth_from_focus_try\cam3d_final\Curved_img_sim_25\curved_img_v2_final.mat')
%% Step 1
% So currently I am checking if the images at different stack can be
% stitched I yes then how
%The first step is to construct an array of stitched images acquired from
%different cameras and stitch them...
%% Basically at each of the 10-20 steps stitch the images and then create a z-stack, 
%Then send the z-stack to the all in focus algorithm to create the depth map and create the final
%camera simulation based on this depth map. 
%....


%% Trying to stitch the images obtained from each camera

% How do you do that? basically what you try to do is calculate the
% homography between the original image and each of the camera...


% So now I will write a new code in the above cam_3d_final folder which
% will register images from each of the camera 9..
%% Simulate image from a 3 x 3 camera module at a specific distance from the curved target...


%% Step 1 read the image and warp it to a surface...
impix = 256;
[I, map] = imread('feat_image_2.tiff'); 
I4=imresize((imread('feat_image_2.tiff')), [impix impix]);
I4=(rgb2gray(I4(:, :, 1:3)));
I4 = im2double(I4);
%Pprs1 = rgb2gray(Pprs3);                        % Grayscale Image
x = -(size(I4,2))*0.5:(size(I4,2))*0.5-1;
y = -(size(I4,2))*0.5:(size(I4,1))*0.5-1;
[X,Y] = meshgrid(x,y);%Create a meshgrid the X, Y coordinates...

Z=-((X).^2+(Y).^2);%%Obtaining the Z-values (Normalising the X, Y value w.r.t the center points
Zmax=max(max(Z));
Zmin=min(min(Z));
%nslices=85; %% Creating the slices, in order to obtain the Bluring with the Z-values.



%Znorm = (Z-Zmin)./(Zmax-Zmin);%Normalising the Z-Value
%Xnorm = (X-min(min(X)))./(max(max(X))-min(min(X)));%Normalising the X -value
%Ynorm = (Y-min(min(Y)))./(max(max(Y))-min(min(Y)));%Normalising the Xq -value


Znorm = (Z-Zmin)./(Zmax-Zmin);%Normalising the Z-Value
Xnorm = (X)./(max(max(X))-min(min(X)));%Normalising the X -value
Ynorm = (Y)./(max(max(Y))-min(min(Y)));%Normalising the Y -value

%% Display warped image
figure(971)
warp(Xnorm, Ynorm, Znorm, I4)%Display the image i.e. wrapped around the curved surface

%% Simulate the cameras 
%Define the x-y coordinates of each camera
% Input fixed parameters.
n1 =5;
ncam = n1*n1;
ov= 0.7;%Overlap bbetween the adjacent cameras.


w1 = 1/((n1-1)*(1-ov)+1);
deltax = (1-ov)*w1;
xcmax = (n1-1)*(deltax/2);
xcmin = -((n1-1)*(deltax/2));

ycmax = (n1-1)*(deltax/2);
ycmin = -((n1-1)*(deltax/2));

xc = linspace(xcmin, xcmax, sqrt(ncam));
yc = linspace(ycmin, ycmax, sqrt(ncam));
[XX, YY] = meshgrid(xc,yc);
%% For each of the nine cameras define the xc, yc, coordinates, focal length, width/height/angle w.r.t. the z-axius
%the reference plane of the camera(zr) and the delthetax, delthetay,
%dell1(Differential change of angle for w, theta and l1)
for i = 1:ncam
    p{i}.xc = XX(i);
    p{i}.yc = YY(i);
    p{i}.f= 0.4;
    p{i}.w = w1;  %original*20
    p{i}.h = w1; 
    p{i}.thetaz = 0;
    p{i}.zr = 2.4;
    p{i}.delthetax = 0.0000001;
    p{i}.delthetay = 0.0000001;
    p{i}.dell1 = 0.0000001;
   

end

nslices = 10; %Ideal value 50
z1ref =  linspace(1.5, 0.75,nslices);


%% Visualize the initial positions of the camera:

thetaxi = 0*ones(1, ncam);%in radians
thetayi = 0*ones(1, ncam); %in radians
l1i = 1*ones(1, ncam); 
for i=1:ncam



%lossf(i) = lossarg(thetaxi(i), thetayi(i), l1i(i), p{i});

% Calculating gradients
%deLthetax(i) = delloss(thetaxi(i), thetayi(i), l1i(i), p{i}, 'thetax');
%deLthetay(i) = delloss(thetaxi(i), thetayi(i), l1i(i), p{i}, 'thetay');
%deLl1(i) = delloss(thetaxi(i), thetayi(i), l1i(i), p{i}, 'l1');

% Defining the surface
%zsurf = @(x, y) (x-125).^2 + (y-125).^2;





v{i} = vparam(thetaxi(i), thetayi(i), l1i(i), p{i});

end


% Plotting cameras before optimization;
for i = 1:ncam
    v1(i, :) = v{i}.v1;
    v2(i, :) = v{i}.v2;
    v3(i, :) = v{i}.v3;
    v4(i, :) = v{i}.v4;
    vh(i, :) = v{i}.vh;
    vr(i, :) = v{i}.vr;
end

hold on
for i = 1:ncam
line([v1(i, 1) vh(i, 1)], [v1(i, 2) vh(i, 2)], [v1(i, 3) vh(i, 3)]);
line([v2(i, 1) vh(i, 1)], [v2(i, 2) vh(i, 2)], [v2(i, 3) vh(i, 3)]);
line([v3(i, 1) vh(i, 1)], [v3(i, 2) vh(i, 2)], [v3(i, 3) vh(i, 3)]);
line([v4(i, 1) vh(i, 1)], [v4(i, 2) vh(i, 2)], [v4(i, 3) vh(i, 3)]);
line([v1(i, 1) v2(i, 1)], [v1(i, 2) v2(i, 2)], [v1(i, 3) v2(i, 3)]);
line([v2(i, 1) v3(i, 1)], [v2(i, 2) v3(i, 2)], [v2(i, 3) v3(i, 3)]);
line([v3(i, 1) v4(i, 1)], [v3(i, 2) v4(i, 2)], [v3(i, 3) v4(i, 3)]);
line([v4(i, 1) v1(i, 1)], [v4(i, 2) v1(i, 2)], [v4(i, 3) v1(i, 3)]);
line([vh(i, 1) vr(i, 1)], [vh(i, 2) vr(i, 2)], [vh(i, 3) vr(i, 3)]);
%patch([t1(1) t2(1) t3(1) t4(1)], [t1(2) t2(2) t3(2) t4(2)], [t1(3) t2(3) t3(3) t4(3)]);
daspect([1 1 1])
axis square

end


hold off;   
%% Starting the loop for capturing the z-stack using flat camera array...
for jkp = 1:nslices




%% Initialize the variables with some values
%Initializing the variables;

zl1i = z1ref(jkp);
thetaxi = 0*ones(1, ncam);%in radians
thetayi = 0*ones(1, ncam); %in radians
l1i = zl1i*ones(1, ncam); 
%% Use lossarg fucntion(Note that we are still not running the gradient descent) 
%in order to obtain the v(i) values, these v{i} values are used to plot the individual spatial configuration 
% of each of the camera.
for i=1:ncam



%lossf(i) = lossarg(thetaxi(i), thetayi(i), l1i(i), p{i});

% Calculating gradients
%deLthetax(i) = delloss(thetaxi(i), thetayi(i), l1i(i), p{i}, 'thetax');
%deLthetay(i) = delloss(thetaxi(i), thetayi(i), l1i(i), p{i}, 'thetay');
%deLl1(i) = delloss(thetaxi(i), thetayi(i), l1i(i), p{i}, 'l1');

% Defining the surface
%zsurf = @(x, y) (x-125).^2 + (y-125).^2;





v{i} = vparam(thetaxi(i), thetayi(i), l1i(i), p{i});

end
%% Camera configuration

%% Storing the v values before the optimization
vbefore = v;
%% Visualize and plot the spatial configurationn of each camera before running the Grad-Descent Optimization
% Plotting cameras before optimization;
for i = 1:ncam
    v1(i, :) = v{i}.v1;
    v2(i, :) = v{i}.v2;
    v3(i, :) = v{i}.v3;
    v4(i, :) = v{i}.v4;
    vh(i, :) = v{i}.vh;
    vr(i, :) = v{i}.vr;
end
%%
hold on
for i = 1:ncam
line([v1(i, 1) vh(i, 1)], [v1(i, 2) vh(i, 2)], [v1(i, 3) vh(i, 3)]);
line([v2(i, 1) vh(i, 1)], [v2(i, 2) vh(i, 2)], [v2(i, 3) vh(i, 3)]);
line([v3(i, 1) vh(i, 1)], [v3(i, 2) vh(i, 2)], [v3(i, 3) vh(i, 3)]);
line([v4(i, 1) vh(i, 1)], [v4(i, 2) vh(i, 2)], [v4(i, 3) vh(i, 3)]);
line([v1(i, 1) v2(i, 1)], [v1(i, 2) v2(i, 2)], [v1(i, 3) v2(i, 3)]);
line([v2(i, 1) v3(i, 1)], [v2(i, 2) v3(i, 2)], [v2(i, 3) v3(i, 3)]);
line([v3(i, 1) v4(i, 1)], [v3(i, 2) v4(i, 2)], [v3(i, 3) v4(i, 3)]);
line([v4(i, 1) v1(i, 1)], [v4(i, 2) v1(i, 2)], [v4(i, 3) v1(i, 3)]);
line([vh(i, 1) vr(i, 1)], [vh(i, 2) vr(i, 2)], [vh(i, 3) vr(i, 3)]);
%patch([t1(1) t2(1) t3(1) t4(1)], [t1(2) t2(2) t3(2) t4(2)], [t1(3) t2(3) t3(3) t4(3)]);
daspect([1 1 1])
axis square

end
hold off;   

   


%% Setting up the overall offset for each camera;
offsetz =0;


%% DISTANCE DEPENDENT BLURRING
% Step 1: The image I4, is blurred by applying the blur kernel over the avergae of
% 4 or 8 pixels section wise in the original image(okay)..

% Step 2: Pass the image into a BBblur function and return a bluured
% image by calculating the perpendicular distance of eachh pixel on the
% curved surface from the reference plane and calculating an average Bblur
% function...



%% Simulate the image from each camera at a particular focal depth and then try aligining and stitching them together..

%% Introducting the vncap: so for each camera the rottrans matrix would be different and dependent on the position of the cameras w.r.t to the surface...


%Calculating the (v4-v1), (v2-v1), (v4-v1)x(v2-v1)
f=0.4;%f = 1 usually;;;
tx = 0.5;%Equalise these two values to XX(i) and YY(i)
ty = 0.5;%These values is not relevant for the modelling below, which involves all the camera
%tz = 1.65;%These values is not relevant for the modelling below, which involves all the camera
%First storing rottrans for each camera before the optimization...
for k = 1:ncam
    vtemp =vbefore{k};
    v1temp = vtemp.v1;
    v2temp = vtemp.v2;
    v3temp = vtemp.v3;
    v4temp = vtemp.v4;
    vxcaptemp = (v3temp-v4temp)./(norm(v3temp-v4temp));
    vycaptemp = (v4temp-v1temp)./(norm(v1temp-v4temp));
    vzcaptemp = -(cross(vxcaptemp, vycaptemp))./(norm(cross(vxcaptemp, vycaptemp)));
    normtemp = -(cross(vxcaptemp, vycaptemp))./(norm(cross(vxcaptemp, vycaptemp)));
    tz = vbefore{k}.vh(3)+offsetz;
    rottranstemp = [vxcaptemp XX(k);vycaptemp YY(k);vzcaptemp tz;0 0 0 1];
    rottransbefore{k} = rottranstemp;
    normlistbefore{k} = normtemp;
    % Creating a new list in which the normlist stores the normal vectors each camera before and after the optimization
    % in form of list...

    %stores the normal values of each image, this will be later on used to calculate the distance between the camera face and the surface ..
    interceptbefore{k} = dot(normlistbefore{k},v1temp);
  
    
    
end





%% Obtaining the list of Blurred images before optimization%%%
%Note this blur is caused due to distance of the focal plane from the
%curved surface
slide = 4;
[Ireturnbefore, dlistbefore]= Bblur3(I4,normlistbefore, interceptbefore, Xnorm(:), Ynorm(:), Znorm(:), ncam, slide);
%Ireturnafter
%List of images acquired from each hof the camera before optimization




%% Now map the 3d surface(The Generated Blurred images) on to the 3d focal plane of each camera
% This would be done for the images before and after the optimization...
%% First Step (BEFORE): Translating the intensities on 3d surface onto the 2d focal plane
%of each camera BEFORE optimization

for i=1:ncam
    
Ireturntemp = Ireturnbefore{i};
lmatrix_fin{i} =  ([Xnorm(:), Ynorm(:), Znorm(:), double(Ireturntemp(:))]); %Stores the X, Y, Z, Intensity in the normalised format for each of the camera
Tcenter{i} = [f 0 0 0;0 f 0 0; 0 0 1 0]; %Intrinsic camera parameters for each camera which basically relates the mapping of the focal plane image on to the camera sensor surface...

xy_center{i} = zeros(size(lmatrix_fin{i}, 1), 4);%% It stores the 2d mapping of the 3d object (in form of the xy coordinats of the camera plane)...

lmatrixtemp = lmatrix_fin{i};
xytemp  = xy_center{i};
for j = 1:size(lmatrix_fin{i}, 1)%No of rows in the lmatrix_fin
    Acent =Tcenter{i}*rottransbefore{i}*[lmatrixtemp(j, 1);lmatrixtemp(j, 2);lmatrixtemp(j, 3);1];
    %Acenttemp{i} = Acent';
    xytemp(j, 1) = Acent(1);
    xytemp(j, 2) = Acent(2);
    xytemp(j, 3) = Acent(3);
    xytemp(j, 4) = lmatrixtemp(j, 4);
    
    
end
xytemp(:, 1) = xytemp(:, 1);%./xytemp(:, 3);
xytemp(:, 2) = xytemp(:, 2);%./xytemp(:, 3);
xy_center{i} = xytemp;
xcam{i} = reshape(xytemp(:, 1), [impix impix]);
ycam{i} = reshape(xytemp(:, 2), [impix impix]);
Icam{i} = reshape(xytemp(:, 4), [impix impix]);
zcam{i} = 2*ones([256 256]);

end


    %Camera Configuration
%  v4<-----w-x-----v3
%  |                |
%  |       .O(0,0)  |l-y
%  |                v
%  v1--------------v2
    
    

%% Check out the camera images by visualizing it 3x3 subplot(Before optimization)
%figure
postemp =  reshape([1:ncam], [sqrt(ncam), sqrt(ncam)]);
postemp =postemp';
for i =1:ncam
subplot(sqrt(ncam), sqrt(ncam), i)
warp(-xcam{postemp(i)}, -ycam{postemp(i)}, zcam{postemp(i)}, Icam{postemp(i)});
view(0, 90)
axis off;
tit = sprintf('Camera %d', i);
title(tit)
end
%%Exchange 7, 3
% Exchange 2, 4
%%Exchange 8, 6



%% Adding the masks
%Note that masking involves converting the vertices of the cameras into 2d
%coordinate and then applying the masks

for ij = 1:1
    v1temp = vbefore{1, ij}.v1;
    v2temp = vbefore{1, ij}.v2;
    v3temp = vbefore{1, ij}.v3;
    v4temp = vbefore{1, ij}.v4;
    
    v1mask = Tcenter{ij}*rottransbefore{ij}*[vbefore{1, ij}.v1(1,1);vbefore{1, ij}.v1(1,2);vbefore{1, ij}.v1(1,3);1];
    v2mask = Tcenter{ij}*rottransbefore{ij}*[vbefore{1, ij}.v2(1,1);vbefore{1, ij}.v2(1,2);vbefore{1, ij}.v2(1,3);1];
    v3mask = Tcenter{ij}*rottransbefore{ij}*[vbefore{1, ij}.v3(1,1);vbefore{1, ij}.v3(1,2);vbefore{1, ij}.v3(1,3);1];
    v4mask = Tcenter{ij}*rottransbefore{ij}*[vbefore{1, ij}.v4(1,1);vbefore{1, ij}.v4(1,2);vbefore{1, ij}.v4(1,3);1];
   
    vbeforemask{1, ij}.v1 = v1mask;
    vbeforemask{1, ij}.v2 = v2mask;
    vbeforemask{1, ij}.v3 = v3mask;
    vbeforemask{1, ij}.v4 = v4mask;
end

vxmin = min([v1mask(1), v2mask(1), v3mask(1), v4mask(1)]);
vxmax = max([v1mask(1), v2mask(1), v3mask(1), v4mask(1)]);

vymin = min([v1mask(2), v2mask(2), v3mask(2), v4mask(2)]);
vymax = max([v1mask(2), v2mask(2), v3mask(2), v4mask(2)]);

%% adding vzmin and vzmax
vzmin = min([v1mask(3), v2mask(3), v3mask(3), v4mask(3)]);
vzmax = min([v1mask(3), v2mask(3), v3mask(3), v4mask(3)]);

%% Now finalising the masked images...

% figure;
% for p =1:9
% subplot(3, 3, p)
% boolx = abs(xcam{postemp(p)})<=abs(vxmax);
% booly = abs(ycam{postemp(p)})<=abs(vymax);
% Itempfinal = Icam{postemp(p)};
% hfig{p} = warp(-xcam{postemp(p)}, -ycam{postemp(p)}, zcam{postemp(p)}, Itempfinal.*boolx.*booly);
% Imbefore{p} = hfig{p}.CData;
% view(0, 90)
% axis off;
% %tit = sprintf('Camera %d', p);
% %title(tit)
% end

%% Trying to stitch the masked images from each camera 

%figure;
for i = 1:ncam
    subplot(sqrt(ncam), sqrt(ncam), i)
    boolxtemp = abs(xcam{postemp(i)})<=(abs(vxmax))*(vzmax)./(f);
    boolytemp = abs(ycam{postemp(i)})<=(abs(vymax))*(vzmax)./(f);
    %boolztemp = abs(ycam{postemp(p)})<=abs(vymax)
    Itempfinal = Icam{postemp(i)};
    Ifinal{postemp(i)} = Itempfinal.*boolxtemp.*boolytemp;
    imshow(Ifinal{postemp(i)});
    axis off;
    tit = sprintf('Camera %d', i);
    title(tit)  
end
Ifinalfinal{jkp} = Ifinal;
end

%% Visualize the initial position of the camera 
%% Visualizing the camera array initial position around the curved surface...


%% (Video Creation) Now create a list of images where different focal z-stacks are simulated and try to visualize them on 9 different cameras...(This will go for the video)
h = figure('Renderer', 'painters', 'Position', [10 10 1500 900]);
a1title = ['Multicam_Defocus_Sim_' num2str(ncam) '_' num2str(ov) '.avi'];
video2 = VideoWriter(a1title, ...
                        'Uncompressed AVI');

video2.FrameRate = 1;
open(video2)

for jkp = 1:nslices
    
    for i = 1:ncam
    subplot(sqrt(ncam), sqrt(ncam), i)
    imshow(Ifinalfinal{1,jkp}{1,postemp(i)})
    tit = sprintf('Camera %d', i);
    tit2 = sprintf('This is slice %d',jkp); 
    title(tit)
    end
    
    
    sgtitle(tit2)
    %pause(0.1)
    F2 = getframe(h);
    writeVideo(video2,F2.cdata);
end
close(video2)
%% Stitching image at each slice(each z-value of the z-stack)...
hBlender1 = vision.AlphaBlender('Operation','Binary mask',...
    'MaskSource','Input port');

height = 256;
width = 256;


%% Defining the warped mask(this mask will be useful for stitching)
warpedMask = zeros(height, width, ncam);

for kk = 1:ncam
    
    warpedMask(:, :, kk) = Ifinalfinal{1,1}{1,postemp(kk)}>0;
    
end

%% Stitching Image at each plane

for jkp = 1:nslices
    
Imfinalstitched{jkp} = 0*ones([height, width]);

for i = 1:ncam
    
    Imfinalstitched{jkp} = step(hBlender1, Imfinalstitched{jkp}, Ifinalfinal{1,jkp}{1,postemp(i)}, warpedMask(:, :, i));
end

end


%% Pass the Stack of images through all in focus algorithm...
%Create Imstack, 
imstackinput  = zeros(width, height, nslices);

for jkp = 1:nslices
imstackinput(:,:,jkp) = Imfinalstitched{jkp};
end


%% Normalising the stitched images at each plane so that all in focus algorithm works well.
stpix =1; %23
epix = 255; %235
for jkp = 1:nslices
    
    imstackoutput(:, :, jkp) = 10*(imstackinput(stpix:epix,stpix:epix,jkp)-min(min(imstackinput(stpix:epix,stpix:epix,jkp))))/(max(max(imstackinput(stpix:epix,stpix:epix,jkp)))-min(min(imstackinput(stpix:epix,stpix:epix,jkp))));
    
    
end


%% (Video Creation) Visualizing the stitched image at each focal plane...(This will go for the video)


h2 = figure('Renderer', 'painters', 'Position', [10 10 400 400]);
a2title = ['Multicam_Defocus_Sim_Stitched_' num2str(ncam) '_' num2str(ov) '.avi'];
video3 = VideoWriter(a2title, ...
                        'Uncompressed AVI');

video3.FrameRate = 1;
open(video3)

for jkp = 3:nslices
imagesc(imstackoutput(:, :, jkp))
tit3 = sprintf("This is image at z1ref = %f",z1ref(jkp));
title(tit3)
colormap('gray')


pause(1)

F3 = getframe(h2);
writeVideo(video3,F3.cdata);

end

close(video3)

%% Visualize the image captured from the flat cam when it grazes the curved surface(note that this is the best possible image obtained from a flatcam configuration for a curved object)

figure;
imagesc(imstackoutput(:, :, 3)) %Note that the value 3 here corresponds to the position at which the flat cam grazes the curved surface, at this point. The flat camera configuration lies just above the curved surface...
tit3 = sprintf("This is image at z1ref = %f",z1ref(3));
title(tit3)
colormap("gray")




%%f
figure;

vxrange = linspace(vxmin, vxmax, epix-stpix);

vyrange = linspace(vymin, vymax, epix-stpix);

[Xrange, Yrange] = meshgrid(vxrange,vyrange);


for jkp = 1:nslices
    imagesc(imstackoutput(:, :, jkp));
    tit_final = sprintf("This is image number %f", z1ref(jkp));
    colormap('gray')
    title(tit_final);
    pause(2);
end


%%

% Passing throughh sff_modified code(Pedro's shape from focus code)...
addpath('C:\Users\COLDD03\OneDrive - Duke University\Data1\DATA\Exp2\Codes\sff');
[z1, r] = sff_modified(imstackoutput);

zc = z1;
zc(r<30) = NaN;

voxel=[3, 3, 3];
zcfiltered= medfilt3(medfilt3(zc));

%% Reshaping the array into one dimension...
kk=1;
for ll=1:size(zcfiltered,1)
    for mm=1:size(zcfiltered, 2)
        if(~isnan(zcfiltered(ll, mm)))
            xtr(1, kk)=ll;
            ytr(1, kk)=mm;
            ztr(1, kk)=zcfiltered(ll, mm);
            kk=kk+1;
        end
    end
end

%% Display the result:

close(gcf), figure
subplot(1,2,1), surf(z1), shading flat, colormap copper
set(gca, 'zdir', 'reverse', 'xtick', [], 'ytick', [])
axis square, grid off, box on
zlabel('pixel depth (mm)')
title('Full depthmap')
view([-0 -90])


subplot(1,2,2), surf(zcfiltered), shading flat, colormap copper
set(gca, 'zdir', 'reverse', 'xtick', [], 'ytick', [])
axis square, grid off, box on
zlabel('pixel depth (mm)')
title('Carved depthmap (R<20dB)(Filtered)')
view([-0 -90])

%% Plotting the 3d scattered points...
deltff = 1000;
[xtrfinal, ytrfinal, ztrfinal]=filter2ztr(xtr, ytr, ztr, deltff);

figure(81);
plot(ztr)

figure(82)

plot(ztrfinal)
title('Median Filtering');
%%also try 3d gaussian triangulation;;
%view the scattered points in 3d;

figure(83)
scatter3(xtrfinal, ytrfinal, ztrfinal'/max(max(ztrfinal')));
title('Scattered Cloud Points');

%% Saving the matrix as XYindicef.mat

XYindicef_final(:, 1) = xtrfinal;
XYindicef_final(:, 2) = ytrfinal;
XYindicef_final(:, 3) = ztrfinal;
save('XYindicef_final.mat','XYindicef_final')



%% Creating the surface from the scattered points.


%% Obtaining and loading the Z-depth map ( that was obbtained by running the all-in-focus algorithm for the z-stack of the images.
%loading the surface obtained from the all infocus algorithm code;;
load('XYindicef_final.mat');
%Sparse sample the input for RBF interpolation to run correctly
%% The part takes in the sparse input for the curved surface and gives out the values corresponding to those areas where 
%input the z value is not present corresponding to the x, y coordinate.
%The goal here is to obtain each hcameras position in space with respect to
%the surface seen by the camera at that particular point...
%In order to get the complete surface you must pass the sparse points into
%the rbf algorithm to get the final surface.
xq = linspace(-0.5, 0.5, 256);
yq = linspace(-0.5, 0.5, 256);
xyshift = 0.5;
[Xq, Yq] = meshgrid(xq, yq);
maxZ = max(max(XYindicef_final(:, 3)));
minZ = min(min(XYindicef_final(:, 3)));
maxY = max(max(XYindicef_final(:, 2)));
minY = min(min(XYindicef_final(:, 2)));
maxX = max(max(XYindicef_final(:, 1)));
minX = min(min(XYindicef_final(:, 1)));

%% Normalising the XYindicef if required (w.r.t. Z, and X, Y also if required)

XYindicef_final(:, 2) = (XYindicef_final(:, 2))/(maxY-minY)-xyshift;
XYindicef_final(:, 1) = (XYindicef_final(:, 1))/(maxX-minX)-xyshift;
XYindicef_final(:, 3) = (XYindicef_final(:, 3)-minZ)/(maxZ-minZ);
%% 
[ZZ, Xyindicesp, op] = fsurfmod(XYindicef_final, Xq, Yq, 1, 1);
% Change the coefficients 1, 1 to some other values,
%fsurfmod is used to plot the 3d surface based on the sparse point that has been gathered from the other - All in Focus algorithm
title("Fitted Surface using RBF interpolation")
%ZZ is the result of Xq, Yq...
%Xq, Yq are the query points and XYindicef are the sparse points obtained
%based on all in focus algorithm...
%% Storing the interpolation coefficients(This will be later on used to get the equation of the surface as seen by a particular
%Surface of the camera.
for i =1:ncam
 p{i}.op = op;
end
%% Plotting the interpolated values of the sparse points, which is used here for visulaization
hold on;
%% Plotting the surface obtained from the RBF interpolation and comparing it against the actual surface(In black meshgrid without any shading interpolation) ...
ZZmax=max(max(ZZ));
ZZmin=min(min(ZZ));
ZZ = (ZZ-ZZmin)./(ZZmax-ZZmin);
figure;
subplot(2, 1, 1)
surf(Xq, Yq, ZZ)
shading interp;
hold on
surf(Xnorm ,Ynorm, Znorm);
title('Fit estimate Curved surface(RBF interpolation) against the original surface');
subplot(2, 1, 2)
RSS = sum((ZZ(:)-Znorm(:)).^2);
TSS = sum((ZZ(:)).^2);
Rsq = 1-(RSS/TSS);
surf(Xq, Yq, ZZ)
tit2 = sprintf("The fit estimate = %f", Rsq);
title(tit2)
shading interp



%% Starting the camera optimization from here...

%% Use lossarg fucntion(Note that we are still not running the gradient descent) 
%in order to obtain the v(i) values, these v{i} values are used to plot the individual spatial configuration 
% of each of the camera.
figure;
for i=1:ncam



lossf(i) = lossarg(thetaxi(i), thetayi(i), l1i(i), p{i});

% Calculating gradients
deLthetax(i) = delloss(thetaxi(i), thetayi(i), l1i(i), p{i}, 'thetax');
deLthetay(i) = delloss(thetaxi(i), thetayi(i), l1i(i), p{i}, 'thetay');
deLl1(i) = delloss(thetaxi(i), thetayi(i), l1i(i), p{i}, 'l1');

% Defining the surface
%zsurf = @(x, y) (x-125).^2 + (y-125).^2;





[~, v{i}] = lossarg(thetaxi(i), thetayi(i), l1i(i), p{i});

end
%% Camera configuration

%% Storing the v values before the optimization
vbefore = v;
%% Visualize and plot the spatial configurationn of each camera before running the Grad-Descent Optimization
% Plotting cameras before optimization;
figure;
for i = 1:ncam
    v1(i, :) = v{i}.v1;
    v2(i, :) = v{i}.v2;
    v3(i, :) = v{i}.v3;
    v4(i, :) = v{i}.v4;
    vh(i, :) = v{i}.vh;
    vr(i, :) = v{i}.vr;
end
%%
hold on
for i = 1:ncam
line([v1(i, 1) vh(i, 1)], [v1(i, 2) vh(i, 2)], [v1(i, 3) vh(i, 3)]);
line([v2(i, 1) vh(i, 1)], [v2(i, 2) vh(i, 2)], [v2(i, 3) vh(i, 3)]);
line([v3(i, 1) vh(i, 1)], [v3(i, 2) vh(i, 2)], [v3(i, 3) vh(i, 3)]);
line([v4(i, 1) vh(i, 1)], [v4(i, 2) vh(i, 2)], [v4(i, 3) vh(i, 3)]);
line([v1(i, 1) v2(i, 1)], [v1(i, 2) v2(i, 2)], [v1(i, 3) v2(i, 3)]);
line([v2(i, 1) v3(i, 1)], [v2(i, 2) v3(i, 2)], [v2(i, 3) v3(i, 3)]);
line([v3(i, 1) v4(i, 1)], [v3(i, 2) v4(i, 2)], [v3(i, 3) v4(i, 3)]);
line([v4(i, 1) v1(i, 1)], [v4(i, 2) v1(i, 2)], [v4(i, 3) v1(i, 3)]);
line([vh(i, 1) vr(i, 1)], [vh(i, 2) vr(i, 2)], [vh(i, 3) vr(i, 3)]);
%patch([t1(1) t2(1) t3(1) t4(1)], [t1(2) t2(2) t3(2) t4(2)], [t1(3) t2(3) t3(3) t4(3)]);
daspect([1 1 1])
axis square

end
hold off;    

%% Running the gradient descent optimization for each camera w.r.t the curved surface it is seeing at a particular time
epochend = 2;
itermax = 10;
losslist = 0;
alpha = 0.002;
%for epochs = 1:epochend
for n= 1:ncam
for epochs =1:epochend
for i = 1:itermax
    losslist(n, i) = lossarg(thetaxi(n), thetayi(n), l1i(n), p{n});
    thetaxi(n) = thetaxi(n) - alpha*delloss(thetaxi(n), thetayi(n), l1i(n), p{n},'thetax');
    thetayi(n) = thetayi(n) - alpha*delloss(thetaxi(n), thetayi(n), l1i(n), p{n},'thetay');
    l1i(n) = l1i(n) - 0.02*alpha*delloss(thetaxi(n), thetayi(n), l1i(n), p{n},'l1');
end
end
end
%end
 %% Plotting cameras after optimization(i.e. extracting the v values after optimization(v contains the vertices informationn about the camera);
 figure;
 for i=1:ncam
     [~, v{i}] = lossarg(thetaxi(i), thetayi(i), l1i(i), p{i});
 end
 %% Storing v values after optimization
 vafter = v;
%% For plotting the second time press here(You will have to close the window befroe doing it)
figure(44)
grid on;
grid minor;
view([30 30]);
surf(Xq, Yq, ZZ);
shading interp
xlabel('X');
ylabel('Y');
zlabel('Z');title('Plotting after optimization');
hold on



for i = 1:ncam
    v1(i, :) = v{i}.v1;
    v2(i, :) = v{i}.v2;
    v3(i, :) = v{i}.v3;
    v4(i, :) = v{i}.v4;
    vh(i, :) = v{i}.vh;
    vr(i, :) = v{i}.vr;
end


for i = 1:ncam
line([v1(i, 1) vh(i, 1)], [v1(i, 2) vh(i, 2)], [v1(i, 3) vh(i, 3)]);
line([v2(i, 1) vh(i, 1)], [v2(i, 2) vh(i, 2)], [v2(i, 3) vh(i, 3)]);
line([v3(i, 1) vh(i, 1)], [v3(i, 2) vh(i, 2)], [v3(i, 3) vh(i, 3)]);
line([v4(i, 1) vh(i, 1)], [v4(i, 2) vh(i, 2)], [v4(i, 3) vh(i, 3)]);
line([v1(i, 1) v2(i, 1)], [v1(i, 2) v2(i, 2)], [v1(i, 3) v2(i, 3)]);
line([v2(i, 1) v3(i, 1)], [v2(i, 2) v3(i, 2)], [v2(i, 3) v3(i, 3)]);
line([v3(i, 1) v4(i, 1)], [v3(i, 2) v4(i, 2)], [v3(i, 3) v4(i, 3)]);
line([v4(i, 1) v1(i, 1)], [v4(i, 2) v1(i, 2)], [v4(i, 3) v1(i, 3)]);
line([vh(i, 1) vr(i, 1)], [vh(i, 2) vr(i, 2)], [vh(i, 3) vr(i, 3)]);
%patch([t1(1) t2(1) t3(1) t4(1)], [t1(2) t2(2) t3(2) t4(2)], [t1(3) t2(3) t3(3) t4(3)]);
daspect([1 1 1])
% axis square

end
hold off;    

%% Setting up the overall offset for each camera;
offsetz =0;





%% Introducting the vncap: so for each camera the rottrans matrix would be different and dependent on the position of the cameras w.r.t to the surface...


%Calculating the (v4-v1), (v2-v1), (v4-v1)x(v2-v1)
f=0.4;%f = 1 usually;;;
tx = 0.5;%Equalise these two values to XX(i) and YY(i)
ty = 0.5;%These values is not relevant for the modelling below, which involves all the camera
tz = 1.65;%These values is not relevant for the modelling below, which involves all the camera
%First storing rottrans for each camera before the optimization...
for k = 1:ncam
    vtemp =vbefore{k};
    v1temp = vtemp.v1;
    v2temp = vtemp.v2;
    v3temp = vtemp.v3;
    v4temp = vtemp.v4;
    vxcaptemp = (v3temp-v4temp)./(norm(v3temp-v4temp));
    vycaptemp = (v4temp-v1temp)./(norm(v1temp-v4temp));
    vzcaptemp = -(cross(vxcaptemp, vycaptemp))./(norm(cross(vxcaptemp, vycaptemp)));
    normtemp = -(cross(vxcaptemp, vycaptemp))./(norm(cross(vxcaptemp, vycaptemp)));
    tz = vbefore{k}.vh(3)+offsetz;
    rottranstemp = [vxcaptemp XX(k);vycaptemp YY(k);vzcaptemp tz;0 0 0 1];
    rottransbefore{k} = rottranstemp;
    normlistbefore{k} = normtemp;
    % Creating a new list in which the normlist stores the normal vectors each camera before and after the optimization
    % in form of list...

    %stores the normal values of each image, this will be later on used to calculate the distance between the camera face and the surface ..
    interceptbefore{k} = dot(normlistbefore{k},v1temp);
  
    
    
end



%% %After rotation and translation(optimization) we have the  vertices and the positions of each camera changed...
for k = 1:ncam
    vtemp =vafter{k};
    v1temp = vtemp.v1;
    v2temp = vtemp.v2;
    v3temp = vtemp.v3;
    v4temp = vtemp.v4;
    vxcaptemp = (v3temp-v4temp)./(norm(v3temp-v4temp));
    vycaptemp = (v4temp-v1temp)./(norm(v1temp-v4temp));
    vzcaptemp = -(cross(vxcaptemp, vycaptemp))./(norm(cross(vxcaptemp, vycaptemp)));
    tz = vafter{k}.vh(3)+offsetz;
    rottranstemp = [vxcaptemp XX(k);vycaptemp YY(k);vzcaptemp tz;0 0 0 1];
    rottransafter{k} = rottranstemp;
    normtemp = -(cross(vxcaptemp, vycaptemp))./(norm(cross(vxcaptemp, vycaptemp)));
    normlistafter{k} = normtemp;
    %stores the normal values of each image, this will be later on used to calculate the distance between the camera face and the surface ..
    interceptafter{k} = dot(normlistafter{k},v1temp);
  
    
    
end




%% DISTANCE DEPENDENT BLURRING
% Step 1: The image I4, is blurred by applying the blur kernel over the avergae of
% 4 or 8 pixels section wise in the original image(okay)..

% Step 2: Pass the image into a BBblur function and return a bluured
% image by calculating the perpendicular distance of eachh pixel on the
% curved surface from the reference plane and calculating an average Bblur
% function...


%% Obtaining the list of Blurred images before optimization%%%
%Note this blur is caused due to distance of the focal plane from the
%curved surface
slide = 4;
[Ireturnbefore, dlistbefore]= Bblur3(I4,normlistbefore, interceptbefore, Xq(:), Yq(:), ZZ(:), ncam, slide);
%Ireturnafter
%List of images acquired from each of the camera before optimization

%% Obtaining the list of Blurred images after optimization%%%
[Ireturnafter, dlistafter]= Bblur3(I4,normlistafter, interceptafter, Xq(:), Yq(:), ZZ(:), ncam, slide);
%Ireturnafter
%List of imagges acquired after the camera optimization



%% Now map the 3d surface(The Generated Blurred images) on to the 3d focal plane of each camera
% This would be done for the images before and after the optimization...
%% First Step (BEFORE): Translating the intensities on 3d surface onto the 2d focal plane
%of each camera BEFORE optimization

for i=1:ncam
    
Ireturntemp = Ireturnbefore{i};
lmatrix_fin{i} =  ([Xnorm(:), Ynorm(:), Znorm(:), double(Ireturntemp(:))]); %Stores the X, Y, Z, Intensity in the normalised format for each of the camera
Tcenter{i} = [f 0 0 0;0 f 0 0; 0 0 1 0]; %Intrinsic camera parameters for each camera which basically relates the mapping of the focal plane image on to the camera sensor surface...

xy_center{i} = zeros(size(lmatrix_fin{i}, 1), 4);%% It stores the 2d mapping of the 3d object (in form of the xy coordinats of the camera plane)...

lmatrixtemp = lmatrix_fin{i};
xytemp  = xy_center{i};
for j = 1:size(lmatrix_fin{i}, 1)%No of rows in the lmatrix_fin
    Acent =Tcenter{i}*rottransbefore{i}*[lmatrixtemp(j, 1);lmatrixtemp(j, 2);lmatrixtemp(j, 3);1];
    %Acenttemp{i} = Acent';
    xytemp(j, 1) = Acent(1);
    xytemp(j, 2) = Acent(2);
    xytemp(j, 3) = Acent(3);
    xytemp(j, 4) = lmatrixtemp(j, 4);
    
    
end
xytemp(:, 1) = xytemp(:, 1);%./xytemp(:, 3);
xytemp(:, 2) = xytemp(:, 2);%./xytemp(:, 3);
xy_center{i} = xytemp;
xcam{i} = reshape(xytemp(:, 1), [impix impix]);
ycam{i} = reshape(xytemp(:, 2), [impix impix]);
Icam{i} = reshape(xytemp(:, 4), [impix impix]);
zcam{i} = 2*ones([256 256]);

end


    %Camera Configuration
%  v4<-----w-x-----v3
%  |                |
%  |       .O(0,0)  |l-y
%  |                v
%  v1--------------v2
    
    

%% Check out the camera images by visualizing it nxn subplot(Before optimization)
figure;
postemp =  reshape([1:ncam], [sqrt(ncam), sqrt(ncam)]);
postemp =postemp';
for i =1:ncam
subplot(sqrt(ncam), sqrt(ncam), i)
warp(-xcam{postemp(i)}, -ycam{postemp(i)}, zcam{postemp(i)}, Icam{postemp(i)});
view(0, 90)
axis off;
tit = sprintf('Camera %d', i);
title(tit)
end
%%Exchange 7, 3
% Exchange 2, 4
%%Exchange 8, 6



%% Second Step(AFTER): Translating the intensities on 3d surface onto the 2d focal planes
%of each camera AFTER optimization


for i=1:ncam
Ireturntemp = Ireturnafter{i};
lmatrix_fin{i} =  ([Xnorm(:), Ynorm(:), Znorm(:), double(Ireturntemp(:))]);
Tcenter{i} = [f 0 0 0;0 f 0 0; 0 0 1 0]; %Intrincis camera parameter

xy_center{i} = zeros(size(lmatrix_fin{i}, 1), 4);%% It stores the 2d mapping of the 3d object onto a screen.

lmatrixtemp = lmatrix_fin{i};
xytemp  = xy_center{i};
for j = 1:size(lmatrix_fin{i}, 1)%No of rows in the lmatrix_fin
    Acent =Tcenter{i}*rottransafter{i}*[lmatrixtemp(j, 1);lmatrixtemp(j, 2);lmatrixtemp(j, 3);1];
    %Acenttemp{i} = Acent';
    xytemp(j, 1) = Acent(1);
    xytemp(j, 2) = Acent(2);
    xytemp(j, 3) = Acent(3);
    xytemp(j, 4) = lmatrixtemp(j, 4);
    
    
end
xytemp(:, 1) = xytemp(:, 1);%./xytemp(:, 3);
xytemp(:, 2) = xytemp(:, 2);%./xytemp(:, 3);
xy_center{i} = xytemp;
xcam{i} = reshape(xytemp(:, 1), [impix impix]);
ycam{i} = reshape(xytemp(:, 2), [impix impix]);
Icam{i} = reshape(xytemp(:, 4), [impix impix]);
zcam{i} = 2*ones([256 256]);

end

%% Check out the camera images by visualizing it on 3x3 subplot(After optimization)
figure;
postemp =  reshape([1:ncam], [sqrt(ncam), sqrt(ncam)]);
postemp =postemp';
for i =1:ncam
subplot(sqrt(ncam), sqrt(ncam), i)
warp(-xcam{postemp(i)}, -ycam{postemp(i)}, zcam{postemp(i)}, Icam{postemp(i)});
view(0, 90)
axis off;
tit = sprintf('Camera %d', i);
title(tit)
end
%%Exchange 7, 3
% Exchange 2, 4
%%Exchange 8, 6

    %Camera Configuration
%  v4<-----w-x-----v3
%  |                |
%  |       .O(0,0)  |l-y
%  |                v
%  v1--------------v2
   

%% Draw the initial and final camera positions overlaid on the original curved surface;;;
%% Display camera positions over the surface before optimization
figure(93)
warp(Xnorm, Ynorm, Znorm, I4)%Display the image i.e. wrapped around the curved surface
hold on;
for i = 1:ncam
    v1(i, :) = vbefore{i}.v1;
    v2(i, :) = vbefore{i}.v2;
    v3(i, :) = vbefore{i}.v3;
    v4(i, :) = vbefore{i}.v4;
    vh(i, :) = vbefore{i}.vh;
    vr(i, :) = vbefore{i}.vr;
end


for i = 1:ncam
line([v1(i, 1) vh(i, 1)], [v1(i, 2) vh(i, 2)], [v1(i, 3) vh(i, 3)]);
line([v2(i, 1) vh(i, 1)], [v2(i, 2) vh(i, 2)], [v2(i, 3) vh(i, 3)]);
line([v3(i, 1) vh(i, 1)], [v3(i, 2) vh(i, 2)], [v3(i, 3) vh(i, 3)]);
line([v4(i, 1) vh(i, 1)], [v4(i, 2) vh(i, 2)], [v4(i, 3) vh(i, 3)]);
line([v1(i, 1) v2(i, 1)], [v1(i, 2) v2(i, 2)], [v1(i, 3) v2(i, 3)]);
line([v2(i, 1) v3(i, 1)], [v2(i, 2) v3(i, 2)], [v2(i, 3) v3(i, 3)]);
line([v3(i, 1) v4(i, 1)], [v3(i, 2) v4(i, 2)], [v3(i, 3) v4(i, 3)]);
line([v4(i, 1) v1(i, 1)], [v4(i, 2) v1(i, 2)], [v4(i, 3) v1(i, 3)]);
line([vh(i, 1) vr(i, 1)], [vh(i, 2) vr(i, 2)], [vh(i, 3) vr(i, 3)]);
%patch([t1(1) t2(1) t3(1) t4(1)], [t1(2) t2(2) t3(2) t4(2)], [t1(3) t2(3) t3(3) t4(3)]);
daspect([1 1 1])
% axis square

end
title("Camera Configuration Before Optimization");
hold off;  
%% Display camera positions over the surface after optimization
figure(94)
warp(Xnorm, Ynorm, Znorm, I4)
%Display the image i.e. wrapped around the curved surface
hold on;
for i = 1:ncam
    v1(i, :) = vafter{i}.v1;
    v2(i, :) = vafter{i}.v2;
    v3(i, :) = vafter{i}.v3;
    v4(i, :) = vafter{i}.v4;
    vh(i, :) = vafter{i}.vh;
    vr(i, :) = vafter{i}.vr;
end


for i = 1:ncam
line([v1(i, 1) vh(i, 1)], [v1(i, 2) vh(i, 2)], [v1(i, 3) vh(i, 3)]);
line([v2(i, 1) vh(i, 1)], [v2(i, 2) vh(i, 2)], [v2(i, 3) vh(i, 3)]);
line([v3(i, 1) vh(i, 1)], [v3(i, 2) vh(i, 2)], [v3(i, 3) vh(i, 3)]);
line([v4(i, 1) vh(i, 1)], [v4(i, 2) vh(i, 2)], [v4(i, 3) vh(i, 3)]);
line([v1(i, 1) v2(i, 1)], [v1(i, 2) v2(i, 2)], [v1(i, 3) v2(i, 3)]);
line([v2(i, 1) v3(i, 1)], [v2(i, 2) v3(i, 2)], [v2(i, 3) v3(i, 3)]);
line([v3(i, 1) v4(i, 1)], [v3(i, 2) v4(i, 2)], [v3(i, 3) v4(i, 3)]);
line([v4(i, 1) v1(i, 1)], [v4(i, 2) v1(i, 2)], [v4(i, 3) v1(i, 3)]);
line([vh(i, 1) vr(i, 1)], [vh(i, 2) vr(i, 2)], [vh(i, 3) vr(i, 3)]);
%patch([t1(1) t2(1) t3(1) t4(1)], [t1(2) t2(2) t3(2) t4(2)], [t1(3) t2(3) t3(3) t4(3)]);
daspect([1 1 1])
% axis square

end
title("Camera Configuration After Optimization");
hold off;   

%% The original image that was supposed to be printed on the curved surface...
figure(95)
warp(Xnorm, Ynorm, Znorm, I4)
view(0, 90)

%% Adding the masks
%Note that masking involves converting the vertices of the cameras into 2d
%coordinate and then applying the masks

for ij = 1:1
    v1temp = vbefore{1, ij}.v1;
    v2temp = vbefore{1, ij}.v2;
    v3temp = vbefore{1, ij}.v3;
    v4temp = vbefore{1, ij}.v4;
    
    v1mask = Tcenter{ij}*rottransbefore{ij}*[vbefore{1, ij}.v1(1,1);vbefore{1, ij}.v1(1,2);vbefore{1, ij}.v1(1,3);1];
    v2mask = Tcenter{ij}*rottransbefore{ij}*[vbefore{1, ij}.v2(1,1);vbefore{1, ij}.v2(1,2);vbefore{1, ij}.v2(1,3);1];
    v3mask = Tcenter{ij}*rottransbefore{ij}*[vbefore{1, ij}.v3(1,1);vbefore{1, ij}.v3(1,2);vbefore{1, ij}.v3(1,3);1];
    v4mask = Tcenter{ij}*rottransbefore{ij}*[vbefore{1, ij}.v4(1,1);vbefore{1, ij}.v4(1,2);vbefore{1, ij}.v4(1,3);1];
   
    vbeforemask{1, ij}.v1 = v1mask;
    vbeforemask{1, ij}.v2 = v2mask;
    vbeforemask{1, ij}.v3 = v3mask;
    vbeforemask{1, ij}.v4 = v4mask;
end

vxmin = min([v1mask(1), v2mask(1), v3mask(1), v4mask(1)]);
vxmax = max([v1mask(1), v2mask(1), v3mask(1), v4mask(1)]);

vymin = min([v1mask(2), v2mask(2), v3mask(2), v4mask(2)]);
vymax = max([v1mask(2), v2mask(2), v3mask(2), v4mask(2)]);


%% Now finalising the masked images...

figure;
for i =1:ncam
subplot(sqrt(ncam), sqrt(ncam), i)
boolx = abs(xcam{postemp(i)})<=abs(vxmax);
booly = abs(ycam{postemp(i)})<=abs(vymax);
Itempfinal = Icam{postemp(i)};
hfig{i} = warp(-xcam{postemp(i)}, -ycam{postemp(i)}, zcam{postemp(i)}, Itempfinal.*boolx.*booly);
Imbefore{i} = hfig{i}.CData;
view(0, 90)
axis off;
tit = sprintf('Camera %d', i);
title(tit)
end

%% Trying to stitch the images from each camera 

figure;
for i = 1:ncam
    subplot(sqrt(ncam), sqrt(ncam), i)
    boolxtemp = abs(xcam{postemp(i)})<=abs(vxmax);
    boolytemp = abs(ycam{postemp(i)})<=abs(vymax);
    %boolztemp = abs(ycam{postemp(p)})<=abs(vymax)
    Itempfinal = Icam{postemp(i)};
    imshow(Itempfinal.*boolxtemp.*boolytemp);
    axis off;
    warpedMask2{i} = (Itempfinal.*boolxtemp.*boolytemp) > 0;
    tit = sprintf('Camera %d', i);
    title(tit)  
end

%% The CData showed for each image...
%( Consider these C-data to be tranformed from the tilted 
%camera to a flat camera(w.r.t the central camera)

figure;
for i = 1:ncam
    subplot(sqrt(ncam), sqrt(ncam), i)
    imshow(Imbefore{i})
    tit = sprintf('Camera %d', i);
    title(tit)  
    
    % Here include the stitching code first calculate the tranformation
    % between the individual camera bended image and the flattened image...
    % and then calculate the tranformation between adjacent images...
    
end

%% Registering manually...
% Manually blending the image...
% First define the mask...
% Then define the image final blended image as imfinal...

height = 256;
width = 256;

%% Defining the mask:


imfinal = 0*ones([height, width], 'single');
%This will store the final blended image...
% Define the hblender

hBlender1 = vision.AlphaBlender('Operation', 'Binary mask', ...
    'MaskSource', 'Input port');


% for k = 1:ncam
%     
%     
%     
% end




%% Manually blend all the images

for k = 1:ncam
    %if(sum(sum(warpedMask2{k}))~=0)
    imfinal = step(hBlender1, imfinal, im2single(Icam{postemp(k)}.*warpedMask2{k}), im2single(warpedMask2{k}));
    %end
end


%% Show the final stitched image %This stitched image from the tilted camera 
figure;
imshow(imfinal)

title("Final registered image captured from the curved cameras configuration");


%% LIST OF FUNCTIONS GOES BELOW
%% Writing down the Bblur function 
%Ireturn returns the blurred image, 
%Norm is the normal vector of the reference plane, 
%Intercept is the intercept point of the reference plane...
%Norm will store the normal vector obbtained using the cross of two
%vectors....of the plane this a list of normal vectors corresponding to
%each hfocal plane
%Intercept will store the intercept values of the reference plane...
%Now the next step is to calculate the distance between the reference
%plane(i.e. the focal plane and the surface)
%and the curved surface...
%slide is the dimension of slide x slide window that will be used for
%sliding across the image to get the convolution running...
%x, y, z are the points on the surface, calculate d, i.e. the distance of
%each point on the surface from the focal plane...

function [Ireturn, dlist] = Bblur3(I, norm, intercept, x, y, z, ncam, slide)
lmatrix_temp = [x y z];


for i = 1:ncam
    dlist{i} = distance1(x, y, z, norm{i}, intercept{i});%here normal and intercept signify the positions of each of the focal plane
    %w.r.t. the curved surface %stores the distance of each pixel on the surface from the focal plane corresponding to each camera
    dlistimg{i} =reshape(dlist{i}, size(I));
    Itemp = I;
    dtemp =dlistimg{i};
    
    for m = 1:size(I, 1)-slide
        for n=1:size(I, 2)-slide
            Itemp(m:m+slide, n:n+slide) = imgaussfilt(Itemp(m:m+slide, n:n+slide),mean2(dtemp(m:m+slide, n:n+slide)));%We are applying a sliding window wise blurring to our image..
            %Factor of two to increase the blurring
        end
    end
    
    Ireturn{i} = Itemp;
    
end


%Applying the sliding window approach here



end


%Input image in this case


function [d] = distance1(x, y, z, normcur, interceptcur)
testmatrix = [x y z];
d= abs((sum((normcur.*testmatrix)')'-interceptcur)./norm(normcur));
end

%% %% Equation of the camera plane
function [loss, v] = lossarg(thetax,thetay, l1, p)

zsurf = @(x, y) -0.15 *(x.^2 + y.^2);
xc = p.xc;
yc = p.yc;
f = p.f;
w = p.w;
h = p.h;
thetaz = p.thetaz;
zr = p.zr;
rotmat=  rotz(thetaz)*roty(thetay)*rotx(thetax);

zh = zr-l1;

vr = [xc, yc, zr];
vh = [xc, yc, zh];
v0 = [xc, yc, zh-f];
d1 = v0-vh;
d1=rotmat*d1';
d = sqrt(h.^2 + w.^2);

dv1 = rotmat*[w/2, h/2, -f]';
dv2 = rotmat*[-w/2, h/2, -f]';
dv3 = rotmat*[-w/2, -h/2, -f]';
dv4 = rotmat*[w/2, -h/2, -f]';


v1 = dv1'+vh;
v2 = dv2'+vh;
v3 = dv3'+vh;
v4 = dv4'+vh;
v.v1 = v1;
v.v2 = v2;
v.v3 = v3;
v.v4 = v4;
v.v0 = v0;
v.vh = vh;
v.vr = vr;


vncap = cross(v1 - v3, v2 - v4)/norm(cross(v1 - v3, v2 - v4));
% Defining the camera plane
fcam2 = @(x, y) (vncap(1)*v1(1) + vncap(2)*v1(2) + vncap(3)*v1(3) - (vncap(1)*x + vncap(2)*y))/(vncap(3));

% Now defining the x, y span of the camera surface;
n=30;
delxcap = (v4-v3)/norm(v4-v3);
delycap = (v2-v3)/norm(v2-v3);
delx = norm(v4-v3)/n;
dely = norm(v2-v3)/n;
k= 1;
for i = 1:n+1
    
    for j = 1:n+1
        XYindice(k, :) =  v3 + delxcap.*delx.*(i-1) + delycap.*dely.*(j-1);
        k=k+1;
    end
end

% Defining the Loss function;

op = p.op;
fcamloss = @(x, y) mean(sum((zsurfmod(x, y, op)-fcam2(x, y)).^2));
loss = fcamloss(XYindice(:, 1), XYindice(:, 2));


end



function v = vparam(thetax,thetay, l1, p)


xc = p.xc;
yc = p.yc;
f = p.f;
w = p.w;
h = p.h;
thetaz = p.thetaz;
zr = p.zr;
rotmat=  rotz(thetaz)*roty(thetay)*rotx(thetax);

zh = zr-l1;

vr = [xc, yc, zr];
vh = [xc, yc, zh];
v0 = [xc, yc, zh-f];
d1 = v0-vh;
d1=rotmat*d1';
d = sqrt(h.^2 + w.^2);

dv1 = rotmat*[w/2, h/2, -f]';
dv2 = rotmat*[-w/2, h/2, -f]';
dv3 = rotmat*[-w/2, -h/2, -f]';
dv4 = rotmat*[w/2, -h/2, -f]';


v1 = dv1'+vh;
v2 = dv2'+vh;
v3 = dv3'+vh;
v4 = dv4'+vh;
v.v1 = v1;
v.v2 = v2;
v.v3 = v3;
v.v4 = v4;
v.v0 = v0;
v.vh = vh;
v.vr = vr;


%vncap = cross(v1 - v3, v2 - v4)/norm(cross(v1 - v3, v2 - v4));

end

function[del] =  delloss(thetax, thetay, l1, p,  arg)
delthetax = p.delthetax;
delthetay = p.delthetay;
dell1 = p.dell1;

switch arg
    case 'thetax'
        del= (lossarg(thetax+delthetax, thetay, l1, p) - lossarg(thetax-delthetax, thetay, l1, p))/(2*delthetax);
    case 'thetay'
        del= (lossarg(thetax, thetay+delthetay, l1, p) - lossarg(thetax, thetay-delthetay, l1, p))/(2*delthetay);
    case 'l1'
        del= (lossarg(thetax, thetay, l1+dell1, p) - lossarg(thetax, thetay, l1-dell1, p))/(2*dell1);
    otherwise
        disp('Enter the correct argument');
end
end
function [Zqsurf] = zsurfmod(x,y, op)
Zqsurf = rbfinterp([x'; y'], op);
end


