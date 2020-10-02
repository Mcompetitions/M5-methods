1. Hardware: Intel Core i7-2600K 3.4Ghz, 8 Cores, 16 Gb RAM, GPU is not used
2. Operating system: Windows 10 Version 1809
3. For running distribution part .NET framework 4.7.2 should be installed.

4-5. For training and predicting new data set the following folder structure should be provided:
*root folder*\frc - scripts for forecasting
*root folder*\raw - dataset (calendar.csv, sales_train_evaluation.csv, sell_prices.csv downloaded from Kaggle)
*root folder*\distribution - executable files for distribution part

Stage 1 - LightGBM point forecast
Based on public notebooks for M5 forecasting accuracy by kyakovlev
Run 
1) *root folder*\frc\fe.py
2) *root folder*\frc\train.py
3) *root folder*\frc\frc.py (will take several hours)
Result - 'point_frc1941.csv' point forecasts in *root folder*\frc folder

Stage 2 - Distribution build
IMPORTANT: Works only on Windows OS with installed .NET framework 4.7.2
Run GoodsForecast.MCompetition.Uncertainty.exe in *root folder*\distribution folder. 
Result - 'Distribution.csv' will appear in *root folder*\distribution folder. It is the file for submission.

6-7. File point_frc1941.csv is generated at stage 1.
Stage 2 can be run only after stage 1 (file point_frc1941.csv is used). File Distribution.csv generated at stage 2.
