import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier

#Setup and load data
SCRIPT_DIR = Path(__file__).parent 
PREPARED_DIR = SCRIPT_DIR.parent / 'prepared'
PLOTS_DIR = SCRIPT_DIR.parent / 'results_plots'
PLOTS_DIR.mkdir(exist_ok=True)

print("Loading data...")
try:
    train_df = pd.read_csv(PREPARED_DIR / 'train_preprocessed.csv')
    test_df = pd.read_csv(PREPARED_DIR / 'test_preprocessed.csv')
except FileNotFoundError:
    print("Error: Run process_airbnb.py first!")
    exit()

#Setup Training Data
target = "high_occupancy"
X_train = train_df.drop(columns=[target])
y_train = train_df[target]

#TRAIN
print("Training Random Forest for Optimization...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

#Test house: randomly select a 'test house' from TEST data 
#Property #5 will be our test house
row_index = 5
sample_house = test_df.iloc[[row_index]].drop(columns=[target]).copy()

print(f"\nOptimizing Price for Property #{row_index}:")
print(f"- Bedrooms: {sample_house['bedrooms'].values[0]}")
# Handle city check safely
is_big_bear = sample_house.get('city_Big Bear Lake', 0).values[0]
print(f"- City: {'Big Bear' if is_big_bear == 1 else 'Other'}")
print(f"- Current Price: ${sample_house['nightly rate'].values[0]}")

#OPTIMIZATION LOOP: Test prices from $50 to $1500
test_prices = np.arange(50, 1501, 25)
results = []

print("Simulating prices...")
for price in test_prices:
    # Update price
    sample_house['nightly rate'] = price
    
    # Get Probability of High Occupancy (0.0 to 1.0)
    prob_high = rf.predict_proba(sample_house)[0][1]
    prob_low = 1.0 - prob_high
    
    #Expected value calculation
    '''Use a weighted average
    We add weights to penalize the model for incorrect predictions, 
    and prevent it from being "lazy" (when making its predictions) by favoring the majority.

    High Occupancy ~ 24 days/mo
    Low Occupancy ~ 9 days/mo'''
    expected_days = (prob_high * 24.0) + (prob_low * 9.0)
    
    projected_revenue = price * expected_days
    
    results.append({
        'Price': price,
        'Win_Probability': prob_high,
        'Expected_Days': expected_days,
        'Projected_Revenue': projected_revenue
    })

results_df = pd.DataFrame(results)

#Visuals
best_row = results_df.loc[results_df['Projected_Revenue'].idxmax()]
optimal_price = best_row['Price']
max_rev = best_row['Projected_Revenue']
expected_days_at_peak = best_row['Expected_Days']

print(f"\n=== RESULT (Probability Weighted) ===")
print(f"Optimal Nightly Rate: ${optimal_price:.2f}")
print(f"Est. Monthly Revenue: ${max_rev:.2f}")
print(f"Est. Days Booked: {expected_days_at_peak:.1f}")

# Create Dual-Axis Plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Revenue (Green)
sns.lineplot(x='Price', y='Projected_Revenue', data=results_df, ax=ax1, color='green', linewidth=2, label='Revenue')
ax1.set_ylabel('Est. Monthly Revenue ($)', color='green')
ax1.tick_params(axis='y', labelcolor='green')

# Plot Occupancy Probability (Blue dashed)
ax2 = ax1.twinx()
sns.lineplot(x='Price', y='Win_Probability', data=results_df, ax=ax2, color='blue', linestyle='--', alpha=0.5, label='Prob. High Occupancy')
ax2.set_ylabel('Probability of High Occupancy', color='blue')
ax2.tick_params(axis='y', labelcolor='blue')
ax2.set_ylim(0, 1.1)

# Mark Optimal Point
ax1.axvline(optimal_price, color='red', linestyle=':', label=f'Optimal: ${optimal_price}')
plt.title(f'Optimized Pricing Model (Property #{row_index})')
plt.grid(True, alpha=0.3)
plt.tight_layout()

save_path = PLOTS_DIR / 'optimization_curve_weighted.png'
plt.savefig(save_path)
print(f"Chart saved to: {save_path}")