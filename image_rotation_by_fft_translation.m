close all; clear all; clc;

I = im2double(imread('sunflowers.jpg'));
figure,imshow(I)

[M, N, s] = size (I); 

temp=ones(2*M,2*N,s);
temp(floor(M/2):floor(M/2)+M-1,floor(N/2):floor(N/2)+N-1,:)=I;
I=temp; clear temp;

figure,imshow(I)
[M N s]=size(I);
teta = pi/3;
a = tan(teta/2)
b = -sin(teta)

 Nx = ifftshift(-fix(M/2):ceil(M/2)-1); 
 Ny = ifftshift(-fix(N/2):ceil(N/2)-1);
 Ix=zeros(length(Nx),length(Ny),3);
 for t=1:3
     for k=1:M
       Ix(k,:,t)=real(ifft(fftn(I(k,:,t)).*exp(-2*1i*pi*(k-floor(M/2))*Ny*a/N)));
     end
 end
 
 Iy = zeros(size(Ix));
 for t=1:3
     for k=1:N
       Iy(:,k,t)=real(ifft(fft(Ix(:,k,t)).*exp(2*1i*pi*(k-floor(N/2))*Nx*b/N)'));
     end
 end
 If = zeros(size(Iy));
 for t=1:3
     for k=1:M
       If(k,:,t)=real(ifft(fft(Iy(k,:,t)).*exp(-2*1i*pi*(k-floor(M/2))*Ny*a/N)));
     end
 end
     
 

 figure,imshow(Ix);
 figure,imshow(Iy);
 figure,imshow(If);