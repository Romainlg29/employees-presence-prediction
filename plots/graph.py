import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from matplotlib.patches import Patch
from statsmodels.tsa.seasonal import seasonal_decompose

class Graph():
    
    def __init__(self):
        
        self.df = pd.read_csv('/Users/flaviechauvat/Documents/M2/PROJET/employees-presence-prediction/df_venues_processed.csv', sep=';')
        
        # DataFrame processing
        # Date conversion
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%d/%m/%Y')

    def average_by_day_of_week(self):
        
        days_order = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        days_mapping = {'lundi': 'monday', 'mardi': 'tuesday', 'mercredi': 'wednesday', 
                       'jeudi': 'thursday', 'vendredi': 'friday', 'samedi': 'saturday', 'dimanche': 'sunday'}

        # Map French day names to English
        self.df['day_of_week'] = self.df['jour_semaine'].map(days_mapping)
        
        # Calculate average presence per day of the week
        presence_by_day = self.df.groupby('day_of_week')['GLOBAL'].mean().reindex(days_order)

        # Create the bar chart
        fig, ax = plt.subplots(figsize=(12, 7))

        # Different colors for weekdays vs weekend
        colors = ['#2E86AB', '#2E86AB', '#2E86AB', '#2E86AB', '#2E86AB', '#E63946', '#E63946']

        bars = ax.bar(range(len(presence_by_day)), presence_by_day.values, 
                    color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)

        # Add values above the bars
        for i, (day, value) in enumerate(presence_by_day.items()):
            if pd.notna(value):
                ax.text(i, value + 10, f'{value:.0f}', 
                        ha='center', va='bottom', fontsize=12, fontweight='bold')

        # Customization
        ax.set_xlabel('Day of the week', fontsize=14, fontweight='bold')
        ax.set_ylabel('Average presence (number of employees)', fontsize=14, fontweight='bold')
        ax.set_title('Average Presence by Day of the Week', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(range(len(presence_by_day)))
        ax.set_xticklabels(presence_by_day.index, fontsize=12)
        ax.grid(axis='y', alpha=0.3)

        # Add average line
        week_average = self.df[self.df['day_of_week'].isin(days_order[:5])]['GLOBAL'].mean()
        ax.axhline(y=week_average, color='orange', linestyle='--', 
                linewidth=2, label=f'Week average: {week_average:.0f}')

        # Legend
        legend_elements = [
            Patch(facecolor='#2E86AB', edgecolor='black', label='Weekdays'),
            Patch(facecolor='#E63946', edgecolor='black', label='Weekend'),
            plt.Line2D([0], [0], color='orange', linestyle='--', linewidth=2, 
                    label=f'Week average: {week_average:.0f}')
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

        plt.tight_layout()
        plt.show()

        # Display statistics
        print("\n" + "="*60)
        print("STATISTICS - PRESENCE BY DAY OF THE WEEK")
        print("="*60)

        for day in days_order:
            if day in presence_by_day.index and pd.notna(presence_by_day[day]):
                count = len(self.df[self.df['day_of_week'] == day])
                print(f"{day.capitalize():12} : {presence_by_day[day]:6.1f} employees ({count} days)")
            else:
                print(f"{day.capitalize():12} : No data")

        # Calculate week/weekend difference
        if pd.notna(presence_by_day.get('saturday')) or pd.notna(presence_by_day.get('sunday')):
            weekend_values = [v for d, v in presence_by_day.items() if d in ['saturday', 'sunday'] and pd.notna(v)]
            if weekend_values:
                weekend_average = np.mean(weekend_values)
                print(f"\n{'Week average':12} : {week_average:.1f} employees")
                print(f"{'Weekend average':12} : {weekend_average:.1f} employees")
                print(f"{'Difference':12} : {week_average - weekend_average:.1f} employees ({((week_average - weekend_average)/week_average)*100:.1f}% decrease)")
        else:
            print(f"\n{'Week average':12} : {week_average:.1f} employees")
            print("No weekend data available")

        print("="*60)

    def presence_by_date(self): 
        # Plot "GLOBAL" by "Date"
        plt.figure(figsize=(14, 7))

        # Create separate masks
        bridge_mask = self.df["pont.congÃ."] == 1
        holiday_mask = self.df["holiday"] == 1
        public_holiday_mask = self.df["jour_feriÃ."] == 1
        
        # Strike mask - any of these columns being 1 indicates a strike
        strike_mask = ((self.df["autre"] == 1) | 
                      (self.df["Greve_nationale"] == 1) | 
                      (self.df["SNCF"] == 1) | 
                      (self.df["prof_nationale"] == 1))
        
        # Normal days (no special event)
        normal_mask = ~(bridge_mask | holiday_mask | public_holiday_mask | strike_mask)

        # Plot all points as a continuous line
        plt.plot(self.df["Date"], self.df["GLOBAL"], color="lightgray", linewidth=1, alpha=0.5)

        # Plot normal points in blue
        plt.scatter(
            self.df[normal_mask]["Date"],
            self.df[normal_mask]["GLOBAL"],
            color="blue",
            s=30,
            zorder=4,
            alpha=0.6,
            label="Normal day"
        )

        # Highlight public holiday points in gold
        if public_holiday_mask.any():
            plt.scatter(
                self.df[public_holiday_mask]["Date"],
                self.df[public_holiday_mask]["GLOBAL"],
                color="gold",
                s=120,
                zorder=7,
                edgecolors="black",
                linewidth=1.5,
                label="Public Holiday",
                marker="*"
            )

        # Highlight bridge/leave points in orange
        if bridge_mask.any():
            plt.scatter(
                self.df[bridge_mask]["Date"],
                self.df[bridge_mask]["GLOBAL"],
                color="orange",
                s=100,
                zorder=6,
                edgecolors="black",
                linewidth=1.5,
                label="Bridge/Leave",
                marker="s"
            )

        # Highlight holiday points in gray
        if holiday_mask.any():
            plt.scatter(
                self.df[holiday_mask]["Date"],
                self.df[holiday_mask]["GLOBAL"],
                color="gray",
                s=100,
                zorder=6,
                edgecolors="black",
                linewidth=1.5,
                label="Holiday",
                marker="D"
            )

        # Highlight strikes in red
        if strike_mask.any():
            plt.scatter(
                self.df[strike_mask]["Date"],
                self.df[strike_mask]["GLOBAL"],
                color="red",
                s=120,
                zorder=8,
                edgecolors="black",
                linewidth=1.5,
                label="Strike",
                marker="X"
            )

        plt.title("Presence by Date - Special Events", fontsize=16, fontweight='bold')
        plt.xlabel("Date", fontsize=12, fontweight='bold')
        plt.ylabel("Presence (number of employees)", fontsize=12, fontweight='bold')
        plt.xticks(rotation=45)
        plt.legend(loc='best', fontsize=10, framealpha=0.9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def correlation_matrix(self):
        
        # Select only numeric columns for correlation
        numeric_df = self.df.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        correlation_matrix = numeric_df.corr()

        # Plot heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Matrix of Features", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def correlation_matrix_without_departments(self):
        df = self.df.drop(columns=['D1', 'D2', 'D3', 'D4'], errors='ignore')
        # Select only numeric columns for correlation
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Calculate correlation matrix
        correlation_matrix = numeric_df.corr()

        # Plot heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
        plt.title("Correlation Matrix of Features", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def correlation_global_reservation(self):
        # Scatter plot of GLOBAL vs reservation with regression line
        plt.figure(figsize=(10, 6))
        
        # Create scatter plot
        sns.scatterplot(x=self.df['Total_reservations'], y=self.df['GLOBAL'], alpha=0.6, color='blue', label='Data points')
        
        # Add regression line
        from scipy.stats import linregress
        x = self.df['Total_reservations'].values
        y = self.df['GLOBAL'].values
        
        # Remove NaN values
        mask = ~np.isnan(x) & ~np.isnan(y)
        x_clean = x[mask]
        y_clean = y[mask]
        
        # Calculate regression
        slope, intercept, r_value, p_value, std_err = linregress(x_clean, y_clean)
        
        # Plot regression line
        x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, color='red', linewidth=2, label=f'Regression line (R² = {r_value**2:.3f})')
        
        plt.title("Correlation between GLOBAL and Reservation", fontsize=16, fontweight='bold')
        plt.xlabel("Total Reservations", fontsize=12, fontweight='bold')
        plt.ylabel("GLOBAL (Presence)", fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Print correlation statistics
        print(f"\nCorrelation coefficient (R): {r_value:.3f}")
        print(f"R-squared (R²): {r_value**2:.3f}")
        print(f"Slope: {slope:.3f}")
        print(f"Intercept: {intercept:.3f}")
        print(f"P-value: {p_value:.6f}")

    def boxplot_by_day_of_week(self):
        """Boxplot showing distribution of presence by day of week with median, spread, and outliers."""
        
        # French day order
        days_order_fr = ['lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'samedi', 'dimanche']
        days_mapping = {'lundi': 'Monday', 'mardi': 'Tuesday', 'mercredi': 'Wednesday', 
                       'jeudi': 'Thursday', 'vendredi': 'Friday', 'samedi': 'Saturday', 'dimanche': 'Sunday'}
        
        df = self.df.dropna(subset=["GLOBAL", "jour_semaine"]).copy()
        
        # Prepare data in order
        data = []
        labels = []
        for day in days_order_fr:
            vals = df.loc[df["jour_semaine"] == day, "GLOBAL"].values
            if len(vals) > 0:
                data.append(vals)
                labels.append(days_mapping.get(day, day))

        # Create boxplot
        fig, ax = plt.subplots(figsize=(12, 7))
        bp = ax.boxplot(data, labels=labels, showfliers=True, patch_artist=True,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=2),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))
        
        # Color weekdays vs weekend differently
        for i, patch in enumerate(bp['boxes']):
            if i < 5:  # Weekdays
                patch.set_facecolor('#2E86AB')
            else:  # Weekend
                patch.set_facecolor('#E63946')
            patch.set_alpha(0.7)
        
        ax.set_title("GLOBAL Presence Distribution by Day of Week", fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel("Day of week", fontsize=14, fontweight='bold')
        ax.set_ylabel("GLOBAL (Number of employees)", fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def boxplot_weekend_vs_weekday(self) -> None:
        """Strong proof of weekend drop: compare distributions."""
        self._validate_cols(["GLOBAL", "is_weekend"])

        df = self.df.dropna(subset=["GLOBAL", "is_weekend"]).copy()

        weekday = df.loc[~df["is_weekend"], "GLOBAL"].values
        weekend = df.loc[df["is_weekend"], "GLOBAL"].values

        plt.figure()
        plt.boxplot([weekday, weekend], labels=["Weekday", "Weekend"], showfliers=True)
        plt.title("GLOBAL presence: Weekday vs Weekend")
        plt.ylabel("GLOBAL")
        plt.tight_layout()
        plt.show()

    def heatmap_month_by_day(self, agg: str = "mean"):
        """
        Monthly aggregation heatmap showing presence patterns across days of the week.
        """
        # French day order
        days_order_fr = ['lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'samedi', 'dimanche']
        days_mapping = {'lundi': 'Monday', 'mardi': 'Tuesday', 'mercredi': 'Wednesday', 
                       'jeudi': 'Thursday', 'vendredi': 'Friday', 'samedi': 'Saturday', 'dimanche': 'Sunday'}

        df = self.df.dropna(subset=["Annee_et_Semaine", "jour_semaine", "GLOBAL"]).copy()
        
        # Create categorical column for proper ordering
        df["jour_semaine_cat"] = pd.Categorical(
            df["jour_semaine"], 
            categories=days_order_fr, 
            ordered=True
        )

        if agg not in ("mean", "sum", "median"):
            raise ValueError("agg must be one of: 'mean', 'sum', 'median'")

        # Create single figure
        plt.figure(figsize=(14, 8))
        
        # Monthly aggregation
        df['Year_Month'] = df['Date'].dt.to_period('M').astype(str)
        
        pivot_monthly = pd.pivot_table(
            df,
            index="Year_Month",
            columns="jour_semaine_cat",
            values="GLOBAL",
            aggfunc=agg
        )
        
        pivot_monthly = pivot_monthly.reindex(columns=[c for c in days_order_fr if c in pivot_monthly.columns])
        pivot_monthly.columns = [days_mapping.get(str(col), str(col)) for col in pivot_monthly.columns]

        # Store original values for statistics
        pivot_original = pivot_monthly.copy()
        
        # Normalize the data (min-max normalization: scale to 0-1)
        pivot_normalized = (pivot_monthly - pivot_monthly.min().min()) / (pivot_monthly.max().max() - pivot_monthly.min().min())
        
        # Plot monthly heatmap with normalized data
        sns.heatmap(pivot_normalized, cmap="YlOrRd", cbar_kws={'label': 'Normalized Presence (0-1)'}, 
                   linewidths=1, linecolor='white', annot=True, fmt='.2f',
                   vmin=0, vmax=1, annot_kws={'size': 9})
        
        plt.title(f"Monthly Aggregation - Normalized Presence Pattern ({agg.capitalize()})", 
                 fontsize=14, fontweight='bold', pad=15)
        plt.xlabel("Day of week", fontsize=11, fontweight='bold')
        plt.ylabel("Month", fontsize=11, fontweight='bold')
        plt.tick_params(axis='y', rotation=0)
        
        plt.tight_layout()
        plt.show()


    def show_days_where_autre_is_one(
        self,
        cols_to_show=None,
        sort_by_date: bool = True,
        max_rows: int | None = None
    ) -> pd.DataFrame:

        # Coerce autre to numeric 0/1

        flagged = self.df[self.df["autre"] == 1].copy()

        if sort_by_date:
            flagged = flagged.sort_values("Date")

        if cols_to_show is None:
            # Useful default columns if they exist
            base = ["Date", "jour_semaine", "GLOBAL", "autre"]
            cols_to_show = [c for c in base if c in flagged.columns]

        flagged_view = flagged[cols_to_show]

        if max_rows is not None:
            flagged_view = flagged_view.head(max_rows)

        print(f"Days where autre == 1: {len(flagged)}")
        print(flagged_view.to_string(index=False))

        return flagged_view

    def correlation_global_temp(self):
        # Scatter plot of GLOBAL vs temperature with regression line
        plt.figure(figsize=(10, 6))
        
        # Create scatter plot
        sns.scatterplot(x=self.df['Temp'], y=self.df['GLOBAL'], alpha=0.6, color='blue', label='Data points')
        
        # Add regression line
        from scipy.stats import linregress
        x = self.df['Temp'].values
        y = self.df['GLOBAL'].values
        
        # Remove NaN values
        mask = ~np.isnan(x) & ~np.isnan(y)
        x_clean = x[mask]
        y_clean = y[mask]
        
        # Calculate regression
        slope, intercept, r_value, p_value, std_err = linregress(x_clean, y_clean)
        
        # Plot regression line
        x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, color='red', linewidth=2, label=f'Regression line (R² = {r_value**2:.3f})')
        
        plt.title("Correlation between GLOBAL and Temperature", fontsize=16, fontweight='bold')
        plt.xlabel("Temperature", fontsize=12, fontweight='bold')
        plt.ylabel("GLOBAL (Presence)", fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Print correlation statistics
        print(f"\nCorrelation coefficient (R): {r_value:.3f}")
        print(f"R-squared (R²): {r_value**2:.3f}")
        print(f"Slope: {slope:.3f}")
        print(f"Intercept: {intercept:.3f}")
        print(f"P-value: {p_value:.6f}")

    def correlation_global_pluie(self): 
        # Scatter plot of GLOBAL vs pluie with regression line
        plt.figure(figsize=(10, 6))
        
        # Create scatter plot
        sns.scatterplot(x=self.df['pluie'], y=self.df['GLOBAL'], alpha=0.6, color='blue', label='Data points')
        
        # Add regression line
        from scipy.stats import linregress
        x = self.df['pluie'].values
        y = self.df['GLOBAL'].values
        
        # Remove NaN values
        mask = ~np.isnan(x) & ~np.isnan(y)
        x_clean = x[mask]
        y_clean = y[mask]
        
        # Calculate regression
        slope, intercept, r_value, p_value, std_err = linregress(x_clean, y_clean)
        
        # Plot regression line
        x_line = np.linspace(x_clean.min(), x_clean.max(), 100)
        y_line = slope * x_line + intercept
        plt.plot(x_line, y_line, color='red', linewidth=2, label=f'Regression line (R² = {r_value**2:.3f})')
        
        plt.title("Correlation between GLOBAL and Rainfall", fontsize=16, fontweight='bold')
        plt.xlabel("Rainfall", fontsize=12, fontweight='bold')
        plt.ylabel("GLOBAL (Presence)", fontsize=12, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # Print correlation statistics
        print(f"\nCorrelation coefficient (R): {r_value:.3f}")
        print(f"R-squared (R²): {r_value**2:.3f}")
        print(f"Slope: {slope:.3f}")
        print(f"Intercept: {intercept:.3f}")
        print(f"P-value: {p_value:.6f}")