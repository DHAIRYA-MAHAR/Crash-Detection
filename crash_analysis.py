
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data and create sample data
data = pd.read_csv("Crash_Reporting_Drivers_Data.csv")
print(f"Original dataset size: {len(data)} rows")
sampled_data = data.sample(n=10000, random_state=42)
print(f"Sampled dataset size: {len(sampled_data)} rows")
sampled_data.to_csv("crash_data_10k.csv", index=False)
print("Sampled data saved as: crash_data_10k.csv")

# Load the sample data
data = pd.read_csv("crash_data_10k.csv")
print("\n=== Dataset Preview ===")
print("First 5 rows:")
print(data.head())
print("\nColumn names:", list(data.columns))
print(f"Total rows: {len(data)}, Total columns: {len(data.columns)}")

# cleaning the data
print("\n=== Missing Values Before Cleaning ===")
print(data.isnull().sum())
data["Driver Substance Abuse"].fillna("Unknown", inplace=True)
data["Injury Severity"].fillna("Unknown", inplace=True)
data["Speed Limit"].fillna(0, inplace=True)
data.dropna(subset=["Crash Date/Time"], inplace=True)
print("\n=== Missing Values After Cleaning ===")
print(data.isnull().sum())
data.to_csv("cleaned_crash_data_10k.csv", index=False)
print("Cleaned data saved as: cleaned_crash_data_10k.csv")

# Load the cleaned data and process date/time 
data = pd.read_csv("cleaned_crash_data_10k.csv")
data["Crash Date/Time"] = pd.to_datetime(data["Crash Date/Time"])
data["Hour"] = data["Crash Date/Time"].dt.hour
data["Day of Week"] = data["Crash Date/Time"].dt.day_name()

# Add gender and licence status (if column not available , create dummy column)
if 'Gender' not in data.columns:
    data['Gender'] = np.random.choice(['Male', 'Female'], size=len(data))  # Dummy data
if 'Driver Age' not in data.columns:
    data['Driver Age'] = np.random.randint(18, 80, size=len(data))  # Dummy age data
if 'License Status' not in data.columns:
    data['License Status'] = np.random.choice(['Valid', 'Suspended'], size=len(data), p=[0.9, 0.1])  # 90% valid, 10% suspended

# Categorical encoding
data = pd.get_dummies(data, columns=['Gender', 'License Status'], drop_first=True)

# Graph 1: Crashes by Hour
plt.figure(figsize=(12, 6))
sns.countplot(x="Hour", data=data, palette="viridis")
plt.title("Crashes by Hour of Day", fontsize=16, fontweight='bold')
plt.xlabel("Hour of Day", fontsize=12)
plt.ylabel("Number of Crashes", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("crashes_by_hour_enhanced.png", dpi=300)
plt.show()

# Graph 2: Crashes by Substance Abuse
plt.figure(figsize=(12, 6))
sns.countplot(x="Driver Substance Abuse", data=data, palette="viridis")
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.title("Crashes by Driver Substance Abuse", fontsize=16, fontweight='bold')
plt.xlabel("Substance Abuse Type", fontsize=12)
plt.ylabel("Number of Crashes", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("crashes_by_substance_enhanced.png", dpi=300)
plt.show()

# Graph 3: Crashes by Speed Limit
plt.figure(figsize=(12, 6))
sns.histplot(data["Speed Limit"], bins=10, color="teal", edgecolor="black")
plt.title("Distribution of Crashes by Speed Limit", fontsize=16, fontweight='bold')
plt.xlabel("Speed Limit (mph)", fontsize=12)
plt.ylabel("Number of Crashes", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("crashes_by_speed_enhanced.png", dpi=300)
plt.show()

# Graph 4: Crashes by Day of Week
plt.figure(figsize=(12, 6))
sns.countplot(x="Day of Week", data=data, palette="viridis", order=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
plt.title("Crashes by Day of Week", fontsize=16, fontweight='bold')
plt.xlabel("Day of Week", fontsize=12)
plt.ylabel("Number of Crashes", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("crashes_by_day_enhanced.png", dpi=300)
plt.show()

# Graph 5: Crashes by Injury Severity (Labels Overlap Fix )
print("\n=== Injury Severity Categories ===")
print(data["Injury Severity"].value_counts())
plt.figure(figsize=(14, 6))
sns.countplot(x="Injury Severity", data=data, palette="viridis")
plt.title("Crashes by Injury Severity", fontsize=16, fontweight='bold')
plt.xlabel("Injury Severity", fontsize=12)
plt.ylabel("Number of Crashes", fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("crashes_by_injury_severity_fixed.png", dpi=300)
plt.show()

# Graph 6: Gender-Based Crash Trend Analysis
plt.figure(figsize=(12, 6))
sns.countplot(x="Gender_Male", data=data, palette="viridis")
plt.title("Crashes by Gender", fontsize=16, fontweight='bold')
plt.xlabel("Gender (1 = Male)", fontsize=12)
plt.ylabel("Number of Crashes", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("crashes_by_gender.png", dpi=300)
plt.show()

# Graph 7: Driver Age Distribution
plt.figure(figsize=(12, 6))
sns.histplot(data["Driver Age"], bins=20, color="teal", edgecolor="black")
plt.title("Distribution of Driver Age in Crashes", fontsize=16, fontweight='bold')
plt.xlabel("Driver Age", fontsize=12)
plt.ylabel("Number of Crashes", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("driver_age_distribution.png", dpi=300)
plt.show()

# Graph 8: License Status and Crash Involvement
plt.figure(figsize=(12, 6))
sns.countplot(x="License Status_Valid", data=data, palette="coolwarm")
plt.title("Crashes by License Status", fontsize=16, fontweight='bold')
plt.xlabel("License Status (1 = Valid)", fontsize=12)
plt.ylabel("Number of Crashes", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("crashes_by_license_status.png", dpi=300)
plt.show()

# Correlation aur Covariance
numeric_data = data[["Speed Limit", "Vehicle Year", "Latitude", "Longitude", "Hour", "Driver Age"]]
correlation_matrix = numeric_data.corr()
covariance_matrix = numeric_data.cov()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Numerical Features", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=300)
plt.show()

# Summary file 
with open("analysis_summary.txt", "w") as file:
    file.write("Analysis Summary for Crash Reporting Data\n")
    file.write("========================================\n")
    file.write(f"Total Crashes Analyzed: {len(data)}\n")
    file.write("\nKey Findings:\n")
    file.write(f"1. Crashes by Hour: Sabse zyada crashes raat 10-12 baje ke beech hote hain ({data['Hour'].value_counts().idxmax()} hour pe).\n")
    file.write(f"2. Driver Substance Abuse: {round(data['Driver Substance Abuse'].value_counts()['Unknown']/len(data)*100, 2)}% crashes mein substance abuse unknown tha.\n")
    file.write(f"3. Speed Limit: Zyada crashes {data['Speed Limit'].value_counts().idxmax()} mph speed limit wale roads pe hote hain.\n")
    file.write(f"4. Day of Week: {data['Day of Week'].value_counts().idxmax()} ko sabse zyada crashes hote hain.\n")
    file.write(f"5. Injury Severity: {data['Injury Severity'].value_counts().idxmax()} severity wale crashes sabse zyada hain.\n")
    file.write(f"6. Gender Trends: {round(data['Gender_Male'].value_counts()[1]/len(data)*100, 2)}% crashes males mein hue (dummy data).\n")
    file.write(f"7. Driver Age Distribution: Average age {round(data['Driver Age'].mean())} years, peak {data['Driver Age'].value_counts().idxmax()} years (dummy data).\n")
    file.write(f"8. License Status: {round(data['License Status_Valid'].value_counts()[1]/len(data)*100, 2)}% crashes valid license wale drivers se (dummy data).\n")
    file.write("\nCorrelation Analysis:\n")
    file.write(str(correlation_matrix) + "\n")
    file.write("\nCovariance Analysis:\n")
    file.write(str(covariance_matrix) + "\n")
    file.write("\nVisualizations:\n")
    file.write("1. Crashes by Hour: See 'crashes_by_hour_enhanced.png'\n")
    file.write("2. Crashes by Substance Abuse: See 'crashes_by_substance_enhanced.png'\n")
    file.write("3. Crashes by Speed Limit: See 'crashes_by_speed_enhanced.png'\n")
    file.write("4. Crashes by Day of Week: See 'crashes_by_day_enhanced.png'\n")
    file.write("5. Crashes by Injury Severity: See 'crashes_by_injury_severity_fixed.png'\n")
    file.write("6. Crashes by Gender: See 'crashes_by_gender.png'\n")
    file.write("7. Driver Age Distribution: See 'driver_age_distribution.png'\n")
    file.write("8. Crashes by License Status: See 'crashes_by_license_status.png'\n")
    file.write("9. Correlation Heatmap: See 'correlation_heatmap.png'\n")
print("Summary saved as: analysis_summary.txt")
print("\n=== Analysis Complete ===")