import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from textblob import TextBlob
import re


# Step 1: Load the CSV file
file_path = 'Juventus Stock Price Monthly.csv'  # Replace with your actual file path
juve_monthly_data = pd.read_csv(file_path)

# Step 2: Convert 'Date' column to datetime format
juve_monthly_data['Date'] = pd.to_datetime(juve_monthly_data['Date'], format='%m/%d/%Y')

# Step 3: Clean 'Vol.' column by converting "M" to numeric
juve_monthly_data['Vol.'] = juve_monthly_data['Vol.'].replace({'M': 'e6'}, regex=True).astype(float)

# Step 4: Clean 'Change %' column by removing '%' and converting to float
juve_monthly_data['Change %'] = juve_monthly_data['Change %'].str.replace('%', '').astype(float)

# Step 5: Sort the data by date in ascending order
juve_monthly_data = juve_monthly_data.sort_values(by='Date')

# Step 6: Calculate percentage changes in price (monthly returns)
juve_monthly_data['Monthly Return'] = juve_monthly_data['Price'].pct_change() * 100

# Step 7: Save the cleaned dataset to an Excel file
output_file = "Juventus_Cleaned_Monthly_Data.xlsx"
juve_monthly_data.to_excel(output_file, index=False)
print(f"Cleaned monthly data saved as '{output_file}'.")

# Step 8: Display the first few rows of the cleaned data
print(juve_monthly_data.head())

import matplotlib.pyplot as plt

# Plot monthly stock price
plt.figure(figsize=(10, 6))
plt.plot(juve_monthly_data['Date'], juve_monthly_data['Price'], label='Price')
plt.title('Juventus Stock Price Over Time')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()

# Plot monthly returns
plt.figure(figsize=(10, 6))
plt.bar(juve_monthly_data['Date'], juve_monthly_data['Monthly Return'], color='orange')
plt.title('Juventus Monthly Returns')
plt.xlabel('Date')
plt.ylabel('Monthly Return (%)')
plt.grid()
plt.show()


# Calculate mean return and volatility
mean_return = juve_monthly_data['Monthly Return'].mean()
monthly_volatility = juve_monthly_data['Monthly Return'].std()

# Annualized volatility (assuming 12 months in a year)
annual_volatility = monthly_volatility * (12 ** 0.5)

print(f"Mean Monthly Return: {mean_return:.2f}%")
print(f"Monthly Volatility: {monthly_volatility:.2f}%")
print(f"Annualized Volatility: {annual_volatility:.2f}%")

# Re-importing necessary libraries after reset
import pandas as pd
import matplotlib.pyplot as plt

# Assuming the dataset is loaded, reloading the data for volatility analysis
#file_path = '/mnt/data/Juventus_Cleaned_Monthly_Data.xlsx'
#juve_monthly_data = pd.read_excel(file_path)

# Extract the necessary data for plotting
dates = juve_monthly_data['Date']
monthly_volatility = juve_monthly_data['Monthly Return'].rolling(window=3).std()  # Rolling 3-month volatility

# Plot the volatility graph
plt.figure(figsize=(12, 6))
plt.plot(dates, monthly_volatility, label="Monthly Volatility", color="orange", linewidth=2)
plt.title("Volatility of Juventus Stock Over Time", fontsize=16)
plt.xlabel("Date", fontsize=12)
plt.ylabel("Volatility (Standard Deviation)", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Prepare data for prediction
juve_monthly_data['Lagged Price'] = juve_monthly_data['Price'].shift(1)  # Lagged feature
juve_monthly_data.dropna(inplace=True)  # Remove rows with NaN due to lagging

X = juve_monthly_data[['Lagged Price']]  # Features
y = juve_monthly_data['Price']  # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print(f"Model Coefficient: {model.coef_[0]:.2f}")
print(f"Model Intercept: {model.intercept_:.2f}")

#Time Series plot

plt.plot(juve_monthly_data['Date'], juve_monthly_data['Price'], label='Actual')
plt.plot(juve_monthly_data['Date'], model.predict(X), label='Predicted')
plt.title("Time Series Prediction")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

import numpy as np
from sklearn.linear_model import LinearRegression

# Assuming 'Price' represents Juventus stock prices and 'Change %' as percent change
# For simplicity, I'll create a simulated market index change data
# Replace `market_change` with actual market index changes if available
juve_monthly_data['Market Change %'] = np.random.uniform(-5, 5, size=len(juve_monthly_data))

# Calculate percent change for Juventus stock (already in 'Change %' column)
juve_stock_change = juve_monthly_data['Change %']

# Calculate Beta using linear regression: y = Beta * x + intercept
X = juve_monthly_data['Market Change %'].values.reshape(-1, 1)  # Market index changes (independent variable)
y = juve_stock_change.values  # Juventus stock changes (dependent variable)

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Extract Beta (coefficient of the independent variable)
beta = model.coef_[0]

print(f"Calculated Beta for Juventus stock: {beta:.4f}")

from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from textblob import TextBlob
import re

# Load Agnelli's speech content
speech_text = """
I can't hide my emotion; a chapter of Juventus' history ends today. We have witnessed changes in ownerships at big clubs such as Chelsea, Milan, Newcastle, Atalanta, and Leeds. Consortiums like Chelsea, RedBird's Milan, PIF's Newcastle, and Pagliuca for Atalanta. We are witnessing an expanding phenomenon such as multi-properties: CityGroup, RedBull, RedBird, 777. We have no certain data, but Goldman & Sachs is investing €1 billion in football; there is interest in the sector. To me, governors didn't provide a suitable answer; they haven't evolved and don't see the difference between game and business. Differences are becoming more and more evident. As a UEFA member and ECA President, the analysis was clear: the system was not sustainable, and clubs were the only ones taking risks. There was disaffection from fans, and we've often been criticized for highlighting this, which is becoming clear also to those managing the 'middle' level. Our proposal was to create an ecosystem for the leading European leagues to increase stability, keeping a balance between national and European competitions. Serie A has had only 68 teams since it became a unique league almost 100 years ago. It's a system open to everyone based on sporting competitiveness. UEFA and ECA proposed it in 2019; then I don't need to remember what happened. There was COVID, and I don't want to touch that part. If I wanted to keep my privileged position as ECA President, I wouldn't have taken certain decisions in 2021. European football needs structural changes; otherwise, we will decline, favoring the Premier League, which will dominate. The current regulators do not want to hear about football's problems. They are in a monopoly position, and I hope the European Court will recognize UEFA's dominant position.
"""

# Clean the text
def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    return text

cleaned_text = clean_text(speech_text)

# Generate Word Cloud
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white", max_words=100).generate(cleaned_text)

# Plot the word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("Word Cloud of Agnelli's Speech", fontsize=16)
plt.show()

# Sentiment Analysis
blob = TextBlob(cleaned_text)
polarity = blob.sentiment.polarity
subjectivity = blob.sentiment.subjectivity

polarity, subjectivity
