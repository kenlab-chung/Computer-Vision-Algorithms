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

**检测步骤**
- 对M、Y通道进行图像反色，突出火焰区域。
- 对反色后的图像，分别进行otsu二值化，并进行边界清理。
- 合并两个二值化图像的重叠区域。
- 通过形态学属性分析，消除异常区域的噪声干扰。
- 提取保留区域范围，进行疑似区域标记。

## 3 基于混沌编码的图像加密算法应用
### 3.1 Logistic 混沌系统
混沌现象是指发生在一个确定性系统中貌似随机的不规则运动，具有不确定性、不可重复、不可预测的行为特点，是非线性动力系统中固有特性。混沌系统一个典型特点是演变过程对初始状态十分敏感。
```
%模拟logistic混沌序列
%研究表名分支参数在[3.56994567,4]时，Logistic映射才具有混沌性质
clear
close all
clc

figure;hold on;box on;
title('Logistic 映射仿真')

xlabel('分支参数\mu')
ylabel('序列\{xn\}')

%分支参数范围0～4，xn范围0～1
axis([0,4,0,1]);
us = linspace(0,4,5e2);
%迭代计算序列xn
for i =1:length(us)
    x = [];
    %设置随机初始值
    x(1) = abs(rand(1));
    u = us(i);
    for n = 1:1.5e2
        %按公式计算
        x(n+1) = u*x(n)*(1-x(n));
    end
    %设置对应绘制的序列
    u = repmat(u,1,length(x));
    plot(u(1e2:1.5e2),x(1e2:1.5e2),'k.','markersize',3);
    %刷新显示
    pause(1e-3);
end
plot([3,3],[0 1],'r--');
plot([3.56994567,3.56994567],[0 1],'g--');

```
运行这段代码，如下图所示，Logistic混沌序列分布区间在(0,1)，其与分支参数&#956;具有明显的关联。当&#956; &le; 3时（红色虚线），迭代结果趋于稳定；然后随着&#956;的增加结果越来越分散；当&#956; &ge; 3.56994567时（绿色虚线），迭代结果不收敛，呈现混沌状态。同时，在分支参数&#956;和初始值确定的情况下，Logistic混沌序列也具有确定性，适合做加密参数。
![image](https://github.com/kenlab-chung/Computer-Vision-Algorithms/assets/59462735/5c8c22d0-bc44-410b-b138-7411be695dd0)



