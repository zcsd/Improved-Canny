#include <stdio.h>
#include <stdlib.h>
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"

#define mode 1// 0 for sobel, 1 for less, 2 for more

using namespace cv;

void createGaussianKernel(int);
void cannyDector();
void useGaussianBlur();
void getGradientImg();
void nonMaxSuppress();
void lessHysteresisThreshold(int, int);
void moreHysteresisThreshold();
Mat combineImage();

Mat oriImage, bluredImage, edgeMagImage, edgeAngImage, thinEdgeImage, thresholdImage;
Mat lowTho, highTho, sobelX, sobelY;
int *gaussianMask, maskRad, maskWidth = 0, maskSum = 0;
float sigma = 0.0, avgGradient = 0.0, var = 0.0;

int main(int argc, char** argv)
{
    Mat combinedImage;
    oriImage = imread("/Users/zichun/Documents/Assignment/CannyEdgeDetector/CED/cell.jpg", 0);
    
    bool isNewSigma = true;
    while (isNewSigma)
    {
        char wndName[] = "Canny Process";
        isNewSigma = false;
        createGaussianKernel(0);
        cannyDector();
        //combine all images for showing
        combinedImage = combineImage();
        if (combinedImage.rows > 600) {
            resize(combinedImage, combinedImage, Size(combinedImage.cols/1.4,combinedImage.rows/1.4));
        }
                
        imshow(wndName, combinedImage);
        waitKey(10);
                
        char tryNewSigma;
        printf("Do you want to try other sigma?(Y/N): ");
        scanf("%s", &tryNewSigma);
        if (tryNewSigma == 'y' || tryNewSigma == 'Y') {
            isNewSigma = true;
            printf("\n-------------Please Try Another Sigma-------------\n");
            destroyWindow(wndName);
            combinedImage.release();
        }
        //release memory
        free(gaussianMask);
        bluredImage.setTo(Scalar(0));
        edgeMagImage.setTo(Scalar(0));
        sobelY.setTo(Scalar(0));
        sobelX.setTo(Scalar(0));
        edgeAngImage.setTo(Scalar(0));
        thinEdgeImage.setTo(Scalar(0));
        thresholdImage.setTo(Scalar(0));
        sigma = 0.0;
        maskRad = 0;
        maskWidth = 0;
        maskSum = 0;
    }
    printf("-------Program End-------\n");
    return 0;
}
//Create Gaussian Kernel.
void createGaussianKernel(int widthType)
{
    printf("Please input standard deviation(>0) and press Enter: ");
    scanf("%f", &sigma);
    if(sigma < 0.01) sigma = 0.01;
    //compute mask width according to sigma value
    if (widthType == 0) {
        //For canny
        maskWidth = int((sigma - 0.01) * 3) * 2 + 1;
    }else if (widthType == 1){
        //for LoG
        maskWidth = 5;
    }
    
    if(maskWidth < 1)   maskWidth = 1;
    printf("Sigma is %.2f, Mask Width is %d.\n", sigma, maskWidth);
    //declare mask as dynamic memory
    gaussianMask = (int*)malloc(maskWidth * maskWidth * sizeof(int));
    
    double gaussianMaskDou[maskWidth][maskWidth], maskMin = 0.0;
    int gaussianMaskInt[maskWidth][maskWidth];
    
    maskRad = maskWidth / 2;
    int i, j;
    //construct the gaussian mask
    for(int x = - maskRad; x <= maskRad; x++)
    {
        for (int y = -maskRad; y <= maskRad; y++)
        {
            i = x + maskRad;
            j = y + maskRad;
            //gaussian 2d function
            gaussianMaskDou[i][j] = exp( (x*x + y*y) / (-2*sigma*sigma) );
            //min value of mask is the first one
            if(i == 0 && j == 0)  maskMin = gaussianMaskDou[0][0];
            //convert mask value double to integer
            gaussianMaskInt[i][j] = cvRound(gaussianMaskDou[i][j] / maskMin);
            maskSum += gaussianMaskInt[i][j];
        }
    }
    
    //printf("Mask Sum is %d, rad is %d.\n", maskSum, maskRad);
    //represent mask using global pointer
    for(i = 0; i <  maskWidth; i++)
        for (j = 0; j < maskWidth; j++)
            *(gaussianMask + i*maskWidth + j) = gaussianMaskInt[i][j];
}

void cannyDector()
{
    useGaussianBlur();
    getGradientImg();
    nonMaxSuppress();
    
    if (mode == 1) {
        int highTh = 0;
        highTh = avgGradient + 1.2 * var;
        printf("low: %d high: %d \n", int(highTh/2), highTh);
        lessHysteresisThreshold(int(highTh/2), highTh);
        //lessHysteresisThreshold(25, 50);
    }else if (mode == 2) {
        moreHysteresisThreshold();
    }else if (mode == 0) {
        lessHysteresisThreshold(32, 64);
    }

}
//For the border, keep the pixel unchanged
void useGaussianBlur()
{
    //keep border pixel unchanged
    bluredImage = oriImage.clone();
    //Convolutuion process, image*mask
    for (int i = 0; i < oriImage.rows; i++)
    {
        for (int j = 0; j < oriImage.cols; j++)
        {
            if ( (i >= maskRad)&&(i <= oriImage.rows-maskRad)&&(j >= maskRad)&&(j<=oriImage.cols-maskRad) )
            {
                double sum = 0;
                
                for (int x = 0; x < maskWidth; x++)
                    for (int y = 0; y < maskWidth; y++)
                    {
                        sum += *(gaussianMask + x*maskWidth + y) * (double)(oriImage.at<uchar>(i + x - maskRad, j + y - maskRad));
                    }
                bluredImage.at<uchar>(i, j) = sum/maskSum;
            }
        }
        
    }
}

void getGradientImg()
{
    edgeMagImage = Mat::zeros(bluredImage.rows, bluredImage.cols, CV_8UC1);
    edgeAngImage = Mat::zeros(bluredImage.rows, bluredImage.cols, CV_8UC1);
    sobelX = Mat::zeros(bluredImage.rows, bluredImage.cols, CV_8UC1);
    sobelY = Mat::zeros(bluredImage.rows, bluredImage.cols, CV_8UC1);
    
    float xMask[3][3],yMask[3][3];
    if (mode == 0) {
        float xxMask[3][3] = { {-1, 0, 1},
            {-2, 0, 2},
            {-1, 0, 1} };
        float yyMask[3][3] = { {1, 2, 1},
            {0, 0 , 0},
            {-1, -2, -1} };
        memcpy(xMask, xxMask, 9*sizeof(float));
        memcpy(yMask, yyMask, 9*sizeof(float));
    }else{
        float xxMask[3][3] = { {-0.3535, 0, 0.3535},
            {-1, 0, 1},
            {-0.3535, 0, 0.3535} };
        float yyMask[3][3] = { {0.3535, 1, 0.3535},
            {0, 0 , 0},
            {-0.3535, -1, -0.3535} };
        memcpy(xMask, xxMask, 9*sizeof(float));
        memcpy(yMask, yyMask, 9*sizeof(float));
    }

    
    int sobelRad = 1;//int(width/2)=3/2=1
    int sobelWidth = 3;
    
    int sumGradient = 0;
    
    for (int i = 0; i < bluredImage.rows; i++)
    {
        for (int j = 0; j < bluredImage.cols; j++)
        {
            if ( i == sobelRad-1 || i == bluredImage.rows-sobelRad || j == sobelRad-1 || j ==bluredImage.cols-sobelRad)
            {
                edgeMagImage.at<uchar>(i, j) = 0;
                edgeAngImage.at<uchar>(i, j) = 255;
                sobelX.at<uchar>(i,j) = 0;
                sobelY.at<uchar>(i,j) = 0;
            }
            else
            {
                int sumX = 0;
                int sumY = 0;
                
                for (int x = 0; x < sobelWidth; x++)
                    for (int y = 0; y < sobelWidth; y++)
                    {
                        sumX += xMask[x][y] * bluredImage.at<uchar>(i+x-sobelRad, j+y-sobelRad);
                        sumY += yMask[x][y] * bluredImage.at<uchar>(i+x-sobelRad, j+y-sobelRad);
                    }
                
                int mag = sqrt(sumX*sumX + sumY*sumY);
                if (mag > 255)  mag = 255;
                edgeMagImage.at<uchar>(i, j) = mag;
                
                sumGradient += mag;
                //Process sobel X
                if (sumX < 0) {
                    if (sumX < -255 || sumX == -255) {
                        sobelX.at<uchar>(i,j) = 255;
                    }else{
                        sobelX.at<uchar>(i,j) = sumX * (-1);
                    }
                }else if (sumX > 255 || sumX == 255){
                    sobelX.at<uchar>(i,j) = 255;
                }else{
                    sobelX.at<uchar>(i,j) = sumX;
                }
                //Process soble Y
                if (sumY < 0) {
                    if (sumY < -255 || sumY == -255) {
                        sobelY.at<uchar>(i,j) = 255;
                    }else{
                        sobelY.at<uchar>(i,j) = sumY * (-1);
                    }
                }else if (sumY > 255 || sumY == 255){
                    sobelY.at<uchar>(i,j) = 255;
                }else{
                    sobelY.at<uchar>(i,j) = sumY;
                }
                
                int ang = (atan2(sumY, sumX)/M_PI) * 180;
                //4 angle, 0 45 90 135
                if ( ( (ang < 22.5) && (ang >= -22.5) ) || (ang >= 157.5) || (ang < -157.5) )
                    ang = 0;
                if ( ( (ang >= 22.5) && (ang < 67.5) ) || ( (ang < -112.5) && (ang >= -157.5) ) )
                    ang = 45;
                if ( ( (ang >= 67.5) && (ang < 112.5) ) || ( (ang < -67.5) && (ang >= -112.5) ) )
                    ang = 90;
                if ( ( (ang >= 112.5) && (ang < 157.5) ) || ( (ang < -22.5) && (ang >= -67.5) ) )
                    ang = 135;
                edgeAngImage.at<uchar>(i, j) = ang;

            }
        }
    }
    
    avgGradient = float(sumGradient) / float(bluredImage.cols * bluredImage.rows);
    printf("average gradient: %.2f\n", avgGradient);
    
    float sumVar = 0;
    
    for (int i = 0; i < bluredImage.rows; i++)
    {
        for (int j = 0; j < bluredImage.cols; j++)
        {
            sumVar += (edgeMagImage.at<uchar>(i,j) -avgGradient) * (edgeMagImage.at<uchar>(i,j) -avgGradient);
        }
    }
    
    var = sqrt(sumVar / (bluredImage.cols * bluredImage.rows));
    printf("average gradient: %.2f\n", var);
}

void nonMaxSuppress()
{
    thinEdgeImage = edgeMagImage.clone();
    
    for (int i = 0; i < thinEdgeImage.rows; i++)
    {
        for (int j = 0; j < thinEdgeImage.cols; j++)
        {
            if ( i == 0 || i == thinEdgeImage.rows-1 || j == 0 || j == thinEdgeImage.cols-1){
                thinEdgeImage.at<uchar>(i, j) = 0;
            }
            else
            {
                //0 degree direction, left and right
                if (edgeAngImage.at<uchar>(i, j) == 0) {
                    if ( edgeMagImage.at<uchar>(i, j) < edgeMagImage.at<uchar>(i, j+1) || edgeMagImage.at<uchar>(i, j) < edgeMagImage.at<uchar>(i, j-1) )
                        thinEdgeImage.at<uchar>(i, j) = 0;
                }
                //45 degree direction,up right and down left
                if (edgeAngImage.at<uchar>(i, j) == 45) {
                    if ( edgeMagImage.at<uchar>(i, j) < edgeMagImage.at<uchar>(i+1, j-1) || edgeMagImage.at<uchar>(i, j) < edgeMagImage.at<uchar>(i-1, j+1) )
                        thinEdgeImage.at<uchar>(i, j) = 0;
                }
                //90 degree direction, up and down
                if (edgeAngImage.at<uchar>(i, j) == 90) {
                    if ( edgeMagImage.at<uchar>(i, j) < edgeMagImage.at<uchar>(i+1, j) || edgeMagImage.at<uchar>(i, j) < edgeMagImage.at<uchar>(i-1, j) )
                        thinEdgeImage.at<uchar>(i, j) = 0;
                }
                //135 degree direction, up left and down right
                if (edgeAngImage.at<uchar>(i, j) == 135) {
                    if ( edgeMagImage.at<uchar>(i, j) < edgeMagImage.at<uchar>(i-1, j-1) || edgeMagImage.at<uchar>(i, j) < edgeMagImage.at<uchar>(i+1, j+1) )
                        thinEdgeImage.at<uchar>(i, j) = 0;
                }
            }
        }
    }
}

void lessHysteresisThreshold(int lowTh, int highTh)
{
    thresholdImage = thinEdgeImage.clone();
    
    for (int i=0; i<thresholdImage.rows; i++)
    {
        for (int j = 0; j<thresholdImage.cols; j++)
        {
            if(thinEdgeImage.at<uchar>(i,j) > highTh)
                thresholdImage.at<uchar>(i,j) = 255;
            else if(thinEdgeImage.at<uchar>(i,j) < lowTh)
                thresholdImage.at<uchar>(i,j) = 0;
            else
            {
                bool isHigher = false;
                bool doConnect = false;
                for (int x=i-1; x < i+2; x++)
                {
                    for (int y = j-1; y<j+2; y++)
                    {
                        if (x <= 0 || y <= 0 || x > thresholdImage.rows || y > thresholdImage.cols)
                            continue;
                        else
                        {
                            if (thinEdgeImage.at<uchar>(x,y) > highTh)
                            {
                                thresholdImage.at<uchar>(i,j) = 255;
                                isHigher = true;
                                break;
                            }
                            else if (thinEdgeImage.at<uchar>(x,y) <= highTh && thinEdgeImage.at<uchar>(x,y) >= lowTh)
                                doConnect = true;
                        }
                    }
                    if (isHigher)    break;
                }
                if (!isHigher && doConnect)
                    for (int x = i-2; x < i+3; x++)
                    {
                        for (int y = j-2; y < j+3; y++)
                        {
                            if (x < 0 || y < 0 || x > thresholdImage.rows || y > thresholdImage.cols)
                                continue;
                            else
                            {
                                if (thinEdgeImage.at<uchar>(x,y) > highTh)
                                {
                                    thresholdImage.at<uchar>(i,j) = 255;
                                    isHigher = true;
                                    break;
                                }
                            }
                        }
                        if (isHigher)    break;
                    }
                if (!isHigher)   thresholdImage.at<uchar>(i,j) = 0;
            }
        }
    }
}

void moreHysteresisThreshold()
{
    lowTho = Mat::zeros(bluredImage.rows, bluredImage.cols, CV_8UC1);
    highTho = Mat::zeros(bluredImage.rows, bluredImage.cols, CV_8UC1);
    Mat avg = Mat::zeros(bluredImage.rows, bluredImage.cols, CV_32FC1);
    Mat var = Mat::zeros(bluredImage.rows, bluredImage.cols, CV_32FC1);

    for (int i = 0; i < bluredImage.rows; i++) {
        for (int j = 0; j < bluredImage.cols; j++) {
            
            float sumGra = 0;
            for (int x = i-10; x < i+11; x++) {
                for (int y = j-10; y < j+11; y++) {
                    float gra;
                    if (x < 0 || y < 0) {
                        gra = 0;
                    }else{
                        gra = edgeMagImage.at<uchar>(x, y);
                    }
                    
                    sumGra += gra;
                }
            }
            avg.at<float>(i,j) = sumGra / float(21*21);
            //printf("%0.2f ", avg.at<float>(i,j));
        }
    }
    
    for (int i = 0; i < bluredImage.rows; i++) {
        for (int j = 0; j < bluredImage.cols; j++) {
            
            float sumVar = 0;
            for (int x = i-10; x < i+11; x++) {
                for (int y = j-10; y < j+11; y++) {
                    float gra;
                    if (x < 0 || y < 0) {
                        gra = 0;
                    }else{
                        gra = edgeMagImage.at<uchar>(x, y);
                    }
                    
                    sumVar += (gra-avg.at<float>(i,j))*(gra-avg.at<float>(i,j));
                }
            }
            var.at<float>(i,j) = sqrt(sumVar / float(21*21));
            //printf("%0.2f ", var.at<float>(i,j));
        }
    }
    
    int lowTh, highTh;
    thresholdImage = thinEdgeImage.clone();
    
    for (int i=0; i<thresholdImage.rows; i++)
    {
        for (int j = 0; j<thresholdImage.cols; j++)
        {
            highTh = int(avg.at<float>(i,j) + 1.1*var.at<float>(i,j));
            lowTh = highTh / 2;
            
            if (thinEdgeImage.at<uchar>(i,j) < int(avg.at<float>(i,j)/5)) {
                thresholdImage.at<uchar>(i,j) = 0;
            }else{ //added
            
            if(thinEdgeImage.at<uchar>(i,j) > highTh)
                thresholdImage.at<uchar>(i,j) = 255;
            else if(thinEdgeImage.at<uchar>(i,j) < lowTh)
                thresholdImage.at<uchar>(i,j) = 0;
            else
            {
                bool isHigher = false;
                bool doConnect = false;
                for (int x=i-1; x < i+2; x++)
                {
                    for (int y = j-1; y<j+2; y++)
                    {
                        if (x <= 0 || y <= 0 || x > thresholdImage.rows || y > thresholdImage.cols)
                            continue;
                        else
                        {
                            if (thinEdgeImage.at<uchar>(x,y) > highTh)
                            {
                                thresholdImage.at<uchar>(i,j) = 255;
                                isHigher = true;
                                break;
                            }
                            else if (thinEdgeImage.at<uchar>(x,y) <= highTh && thinEdgeImage.at<uchar>(x,y) >= lowTh)
                                doConnect = true;
                        }
                    }
                    if (isHigher)    break;
                }
                if (!isHigher && doConnect)
                    for (int x = i-2; x < i+3; x++)
                    {
                        for (int y = j-2; y < j+3; y++)
                        {
                            if (x < 0 || y < 0 || x > thresholdImage.rows || y > thresholdImage.cols)
                                continue;
                            else
                            {
                                if (thinEdgeImage.at<uchar>(x,y) > highTh)
                                {
                                    thresholdImage.at<uchar>(i,j) = 255;
                                    isHigher = true;
                                    break;
                                }
                            }
                        }
                        if (isHigher)    break;
                    }
                if (!isHigher)   thresholdImage.at<uchar>(i,j) = 0;
            }
            }//added
        }
    }
}

Mat combineImage()
{
    Mat h1CombineImage, h2CombineImage, allImage;
    Mat extraImage = Mat(oriImage.rows, oriImage.cols, CV_8UC1, Scalar(255));
    char sigmaChar[10];
    sprintf(sigmaChar, "%.2f", sigma);
    
    putText(extraImage, "Ori, Gaus, Grad, Grad X", Point(10,20), FONT_HERSHEY_PLAIN, 1, Scalar(0));
    putText(extraImage, "NMS, Threshold, White, Grad Y", Point(10,38), FONT_HERSHEY_PLAIN, 1, Scalar(0));
    putText(extraImage, "Sigma: ", Point(10,56), FONT_HERSHEY_PLAIN, 1, Scalar(0));
    putText(extraImage, sigmaChar, Point(65,56), FONT_HERSHEY_PLAIN, 1, Scalar(0));
    
    hconcat(oriImage, bluredImage, h1CombineImage);
    hconcat(h1CombineImage, edgeMagImage, h1CombineImage);
    hconcat(h1CombineImage, sobelY, h1CombineImage);
    hconcat(thinEdgeImage, thresholdImage, h2CombineImage);
    hconcat(h2CombineImage, extraImage, h2CombineImage);
    hconcat(h2CombineImage, sobelX, h2CombineImage);
    vconcat(h1CombineImage, h2CombineImage, allImage);
    
    return allImage;
}
