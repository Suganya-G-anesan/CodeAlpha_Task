import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Unemployment in India.csv")
df.columns = df.columns.str.strip().str.replace(" ", "_")
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["Date"]).sort_values("Date")

print("Rows after cleaning:", len(df))
print("Columns: Region, Date, Frequency, Estimated_Unemployment_Rate_(%), Estimated_Employed, Estimated_Labour_Participation_Rate_(%), Area")

monthly = df.groupby(pd.Grouper(key="Date", freq="M"))["Estimated_Unemployment_Rate_(%)"].mean()

plt.figure(figsize=(14,6))
plt.plot(monthly.index, monthly.values, linewidth=3)
plt.title("Overall Unemployment Trend (2019–2020)")
plt.ylabel("Unemployment Rate (%)")
plt.xlabel("Date")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

covid = df[(df["Date"] >= "2020-03-01") & (df["Date"] <= "2020-06-30")]

plt.figure(figsize=(14,6))
sns.lineplot(data=covid, x="Date", y="Estimated_Unemployment_Rate_(%)", hue="Region", linewidth=2, marker="o")
plt.title("Region-wise Covid Impact (Mar–Jun 2020)")
plt.ylabel("Unemployment Rate (%)")
plt.xlabel("Date")
plt.xticks(rotation=45)
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

max_row = df.loc[df["Estimated_Unemployment_Rate_(%)"].idxmax()]
print("Region:", max_row["Region"], f"({max_row['Area']})")
print("Date:", max_row["Date"].strftime("%B %Y"))
print("Rate:", f"{max_row['Estimated_Unemployment_Rate_(%)']:.2f}%")

min_row = df[df["Estimated_Unemployment_Rate_(%)"] > 0].sort_values("Estimated_Unemployment_Rate_(%)").iloc[0]
print("Region:", min_row["Region"], f"({min_row['Area']})")
print("Date:", min_row["Date"].strftime("%B %Y"))
print("Rate:", f"{min_row['Estimated_Unemployment_Rate_(%)']:.2f}%")

print("Seasonality / Trend Insights:")
print("• Unemployment stable through 2019 with mild seasonal variation.")
print("• Sharp Covid spike in April–May 2020.")
print("• Rural agriculture cycles cause predictable dips and rises.")
print("• Recovery begins after June 2020.")

print("Policy Insights:")
print("• Covid exposes need for stronger social safety nets.")
print("• Migrant-heavy states need better job protection.")
print("• Urban areas need MSME support and digital skilling.")
print("• State-level employment policy more effective than national uniform policy.")
