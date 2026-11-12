clc;
img = (imread('sunflowers.jpg'));
%m = input('Number of pixels by which image needs to be shifted: ');
m=20;
[a b c] = size(img);
for i = a-m:a
    for j = 1:b
        img(i,j,:) = 0;
    end
end

for i = 1:a
    for j = b-m:b
        img(i,j,:) = 0;
    end
end

Y = (fftn(img));
%Y = fftshift(Y);
[M N L] = size(Y);
wM = zeros(M,M);
wN = zeros(N,N);

for x = 0:M-1
    for u = 0:M-1
        wM(x+1,u+1) = exp(-1i*u*2*pi/M*m)*exp(2*pi*1i/M * x*u);
    end
end

for y = 0:N-1
    for v = 0:N-1
        wN(v+1,y+1) = exp(-1i*v*2*pi/N*m)*exp(2*pi*1i/N * y*v);
    end
end

r = ((abs(1/(M*N)*wM * im2double(Y(:,:,1)) * wN)));
g = ((abs(1/(M*N)*wM * im2double(Y(:,:,2)) * wN))); 
b = histeq((abs(1/(M*N)*wM * im2double(Y(:,:,3)) * wN)));
R = ((cat(3,r,g,b)));
figure;
imshow(mat2gray((R)));

%Z = fftn(R);
%Z = fftshift(Z); %shifts to centre
%Z = abs(Z);
%Z = log(Z+1); %scaling
%Z = mat2gray(Z); %scales image between 0 and 1
%figure();
%imshow(Z,[]); 