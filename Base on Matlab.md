# Base on Matlab
## 1 基于图像增强的雾天图像优化方法
single_scale_retinex
```
function enhanced_img = single_scale_retinex(img, sigma)
    % 将图像转换为双精度
    img_double = im2double(img);

    % 高斯滤波进行模糊处理
    img_blurred = imgaussfilt(img_double, sigma);

    % 计算反射分量
    reflection = img_double ./ (img_blurred + eps);

    % 调整反射分量，这里使用对数变换
    reflection_adjusted = log(1 + reflection);

    % 合成增强图像
    enhanced_img = reflection_adjusted .* img_double;
end
```
关键代码
```
clear 
close all
clc

I = imread("wt.jpg");
figure;imshow(I);title('原始图像');
%figure;imhist(I);title('直方图分布');

%直方图均衡化
J = histeq(I);
figure;imshow(J);title('直方图均衡化');

%全局直方图均衡化
for i = 1 : size(I,3)
    J(:,:,i) = histeq(I(:,:,i));
end
figure;imshow(J);title('全局直方图均衡化');

%限制对比度的自适应直方图均衡（CLAHE 增强）
for i = 1 : size(I,3)
    %遍历三通道进行处理
    J(:,:,i) = adapthisteq(I(:,:,i));
end
figure;imshow(J);title('全局直方图均衡化');

%Retinex 增强

I = imread("wt.jpg");
%图像维度
[M,N,~] =size(I);

%滤波器参数
sigma = 150;
%滤波器
g_filter = fspecial("gaussian",[M,N],sigma);
gf_filter = fft2(double(g_filter));
for i= 1:size(I,3)
    %当前通道矩阵
    si = double(I(:,:,i));
    
    %对s进行log计算
    si(si==0) = eps;
    si_log = log(si);
    
    %fft变换
    sif = fft2(si);
    
    %滤波器滤波
    sif_filter = sif.*gf_filter;

    %ifft变换
    sif_filter_i = ifft2(sif_filter);
    sif_filter_i(sif_filter_i==0)=eps;

    %对g*s进行log计算
    sif_filter_log = log(sif_filter_i);

    %计算log(s)-log(g*s)
    Jr = si_log - sif_filter_log;

    %计算exp
    Jr_exp = exp(Jr);

    %归一化
    Jr_exp_min = min(min(Jr_exp));
    Jr_exp_man = max(max(Jr_exp));

    Jr_exp =(Jr_exp-Jr_exp_min)/(Jr_exp_man-Jr_exp_min);

    %合并赋值
    J(:,:,i) = adapthisteq(Jr_exp);
end
figure;imshow(J);title('Retinex 增强');


% 读取图像
img = imread('wt.jpg');

% 设置高斯滤波的尺度参数
sigma = 150;

% 调用 Retinex 算法函数
enhanced_img = single_scale_retinex(img, sigma);

% 显示结果
figure;
subplot(1, 2, 1);
imshow(img);
title('Original Image');

subplot(1, 2, 2);
imshow(enhanced_img);
title('Enhanced Image');
```
## 2 基于颜色特征的森林火情预警识别
### 2.1 火情特征分析
主要是对疑似火焰区检测分割。根据火焰呈现出的颜色特的，选择不同颜色空间进行分析，突出火焰区域，并将其与背景图图像进行分离，进而分割出火情疑似区域。

常用的颜色空间包括：

- RGB：硬件显示设备最常用的颜色模型。
- HSV：面向人类视觉感知的颜色模型，由H、S、V三个通道构成，分别对应：色调（Hue），取值范围为0&deg;～360&deg;；饱和度（Saturation），取值范围0～1值越大，颜色饱和度越高；亮度（Value），取值范围0～1，值越大则越亮。
- CMYK：由C、M、Y、K四通道构成，对应于Cyan、Magenta、Yellow、Black，也称为青色、品红色、黄色、黑色四个分量。广泛应用于印刷业，也常见于日常生活中喷墨打印机的墨盒配件，通过CMYK不同浓度 的叠加来生成不同色彩。

**识别原理：**
选择森林火情样例图进行分析，通过颜色空间转换将图像由RGB颜色空间转到CMYK颜色空间，并按照C、M、Y、K四个通道进行分离，显示各个通道分量图，在CMYK颜色空间中，各个通道均在一定程度上突出火焰区域，并能与背景图产生较为明显的对比度分离效果。考虑到C分量图存在一定模糊性，以及K分量与C、M、Y分量的计算关系，推荐选择M、Y通道分量作为参考图进行综合分析，通过对两个通道分量做图像分割，定位出火焰区域的位置。




