import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta


class WaitTimeVisualizations:
    """
    Advanced visualizations for wait-time analysis.
    Three core visualizations for internship portfolio:
    1. Interactive Wait-Time Heatmap (Hour × Day)
    2. Live Queue Simulation Animation
    3. Queue Stability Phase Diagram (λ vs μ)
    """

    def __init__(self, df):
        self.df = df.copy()
        self._prepare_data()

    def _prepare_data(self):
        if 'x_ScheduledDTTM' in self.df.columns:
            self.df['hour'] = pd.to_datetime(self.df['x_ScheduledDTTM']).dt.hour
            self.df['day_of_week'] = pd.to_datetime(self.df['x_ScheduledDTTM']).dt.dayofweek
            self.df['day_name'] = pd.to_datetime(self.df['x_ScheduledDTTM']).dt.day_name()

        if 'x_ArrivalDTTM' in self.df.columns:
            self.df['arrival_time'] = pd.to_datetime(self.df['x_ArrivalDTTM'])

    def create_heatmap_interactive(self, save_path='wait_time_heatmap.html'):
        """
        1. Interactive Wait-Time Heatmap (Hour × Day)

        Shows: When the clinic becomes congested
        Why it's impressive: Hospital-grade staffing decision visualization
        """
        heatmap_data = self.df.pivot_table(
            values='Wait',
            index='day_of_week',
            columns='hour',
            aggfunc='mean'
        )

        day_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=[f'{h:02d}:00' for h in heatmap_data.columns],
            y=[day_labels[i] for i in heatmap_data.index],
            colorscale='RdYlGn_r',
            colorbar=dict(title='Avg Wait (min)'),
            hovertemplate='Day: %{y}<br>Hour: %{x}<br>Avg Wait: %{z:.1f} min<extra></extra>'
        ))

        fig.update_layout(
            title={
                'text': 'Wait-Time Heatmap: Temporal Demand Signature<br><sub>Hospital-grade congestion analysis</sub>',
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title='Hour of Day',
            yaxis_title='Day of Week',
            width=1200,
            height=600,
            font=dict(size=12),
            plot_bgcolor='white'
        )

        fig.write_html(save_path)
        print(f"Interactive heatmap saved to {save_path}")
        return fig

    def create_heatmap_static(self, save_path='wait_time_heatmap_static.png'):
        """
        Static version of heatmap for presentations
        """
        heatmap_data = self.df.pivot_table(
            values='Wait',
            index='day_of_week',
            columns='hour',
            aggfunc='mean'
        )

        day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

        fig, ax = plt.subplots(figsize=(14, 6))
        sns.heatmap(
            heatmap_data,
            cmap='RdYlGn_r',
            annot=True,
            fmt='.0f',
            cbar_kws={'label': 'Average Wait Time (minutes)'},
            yticklabels=[day_labels[i] for i in heatmap_data.index],
            xticklabels=[f'{h}:00' for h in heatmap_data.columns],
            ax=ax
        )

        ax.set_title('Wait-Time Heatmap: Temporal Demand Signature\nHospital-grade congestion analysis',
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
        ax.set_ylabel('Day of Week', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Static heatmap saved to {save_path}")
        plt.close()
        return fig

    def create_queue_simulation_animation(self, duration_hours=24, save_path='queue_simulation.gif'):
        """
        2. Live Queue Simulation Animation

        Shows: Dynamic evolution of the queue, when λ crosses μ
        Why it's impressive: Demonstrates system modeling expertise
        """
        sample_data = self.df.sort_values('arrival_time').head(1000).copy()

        if 'arrival_time' not in sample_data.columns:
            print("Error: arrival_time column not found")
            return None

        start_time = sample_data['arrival_time'].min()
        sample_data['minutes_elapsed'] = (sample_data['arrival_time'] - start_time).dt.total_seconds() / 60

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))

        time_points = np.arange(0, sample_data['minutes_elapsed'].max(), 5)

        def compute_queue_state(t):
            current_data = sample_data[sample_data['minutes_elapsed'] <= t]
            if len(current_data) == 0:
                return 0, 0, 0, 0

            queue_length = len(current_data[current_data['minutes_elapsed'] > t - 30])
            avg_wait = current_data['Wait'].tail(50).mean() if len(current_data) >= 50 else current_data['Wait'].mean()
            arrival_rate = len(current_data[current_data['minutes_elapsed'] > t - 60]) / 60 if t >= 60 else 0
            service_rate = current_data['mu_per_min'].tail(50).mean() if len(current_data) >= 50 else current_data['mu_per_min'].mean()

            return queue_length, avg_wait, arrival_rate, service_rate

        queue_lengths = []
        wait_times = []
        arrival_rates = []
        service_rates = []

        for t in time_points:
            q, w, arr, srv = compute_queue_state(t)
            queue_lengths.append(q)
            wait_times.append(w)
            arrival_rates.append(arr)
            service_rates.append(srv)

        def animate(frame):
            ax1.clear()
            ax2.clear()
            ax3.clear()

            current_idx = frame

            ax1.plot(time_points[:current_idx+1], queue_lengths[:current_idx+1],
                    color='#2E86AB', linewidth=2.5, label='Queue Length')
            ax1.fill_between(time_points[:current_idx+1], queue_lengths[:current_idx+1],
                            alpha=0.3, color='#2E86AB')
            ax1.set_ylabel('Queue Length (patients)', fontsize=11, fontweight='bold')
            ax1.set_title(f'Live Queue Simulation | Time: {time_points[current_idx]:.0f} minutes',
                         fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left')
            ax1.set_xlim(0, time_points[-1])
            ax1.set_ylim(0, max(queue_lengths) * 1.1 if max(queue_lengths) > 0 else 10)

            ax2.plot(time_points[:current_idx+1], wait_times[:current_idx+1],
                    color='#A23B72', linewidth=2.5, label='Predicted Wait Time')
            ax2.axhline(y=np.mean(wait_times), color='red', linestyle='--',
                       linewidth=1.5, label='Average Wait', alpha=0.7)
            ax2.fill_between(time_points[:current_idx+1], wait_times[:current_idx+1],
                            alpha=0.3, color='#A23B72')
            ax2.set_ylabel('Wait Time (minutes)', fontsize=11, fontweight='bold')
            ax2.set_title('Predicted Wait Time Evolution', fontsize=12, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper left')
            ax2.set_xlim(0, time_points[-1])
            ax2.set_ylim(0, max(wait_times) * 1.1 if max(wait_times) > 0 else 30)

            ax3.plot(time_points[:current_idx+1], arrival_rates[:current_idx+1],
                    color='#F18F01', linewidth=2.5, label='λ (arrival rate)', alpha=0.8)
            ax3.plot(time_points[:current_idx+1], service_rates[:current_idx+1],
                    color='#06A77D', linewidth=2.5, label='μ (service rate)', alpha=0.8)

            if current_idx > 0 and arrival_rates[current_idx] > service_rates[current_idx]:
                ax3.axvspan(time_points[current_idx-1], time_points[current_idx],
                           alpha=0.2, color='red', label='System Unstable (λ > μ)')

            ax3.set_xlabel('Time (minutes)', fontsize=11, fontweight='bold')
            ax3.set_ylabel('Rate (patients/min)', fontsize=11, fontweight='bold')
            ax3.set_title('Arrival vs Service Rate (Stability Analysis)', fontsize=12, fontweight='bold')
            ax3.grid(True, alpha=0.3)
            ax3.legend(loc='upper left')
            ax3.set_xlim(0, time_points[-1])

            plt.tight_layout()

        anim = FuncAnimation(fig, animate, frames=len(time_points), interval=100, repeat=True)

        anim.save(save_path, writer='pillow', fps=10, dpi=100)
        print(f"Queue simulation animation saved to {save_path}")
        plt.close()
        return anim

    def create_phase_diagram(self, save_path='phase_diagram.png'):
        """
        3. Queue Stability Phase Diagram (λ vs μ)

        Shows: Regions where system is stable vs unstable
        Why it's impressive: PhD-level operations research visualization
        """
        if 'lambda_per_min' not in self.df.columns or 'mu_per_min' not in self.df.columns:
            print("Error: lambda_per_min or mu_per_min not found")
            return None

        sample = self.df.sample(n=min(10000, len(self.df)), random_state=42)

        lambda_range = np.linspace(sample['lambda_per_min'].min(),
                                   sample['lambda_per_min'].max(), 50)
        mu_range = np.linspace(sample['mu_per_min'].min(),
                              sample['mu_per_min'].max(), 50)

        Lambda, Mu = np.meshgrid(lambda_range, mu_range)

        def theoretical_wait(lam, mu, servers=1):
            rho = lam / (servers * mu)
            if rho >= 1:
                return 500
            else:
                return (rho / (mu * (1 - rho)))

        Wait = np.vectorize(theoretical_wait)(Lambda, Mu, servers=sample['servers'].mode()[0] if 'servers' in sample.columns else 1)
        Wait = np.clip(Wait, 0, 500)

        fig, ax = plt.subplots(figsize=(12, 10))

        levels = [0, 10, 20, 30, 50, 75, 100, 150, 200, 300, 500]
        contour = ax.contourf(Lambda, Mu, Wait, levels=levels, cmap='RdYlGn_r', alpha=0.8)
        cbar = plt.colorbar(contour, ax=ax, label='Predicted Wait Time (minutes)')

        contour_lines = ax.contour(Lambda, Mu, Wait, levels=levels, colors='black',
                                   linewidths=0.5, alpha=0.3)
        ax.clabel(contour_lines, inline=True, fontsize=8)

        lambda_line = np.linspace(min(lambda_range), max(lambda_range), 100)
        mu_line = lambda_line
        ax.plot(lambda_line, mu_line, 'k--', linewidth=3, label='λ = μ (Stability Boundary)', zorder=10)

        unstable_region = ax.fill_between(lambda_range, lambda_range, max(mu_range),
                                          alpha=0.15, color='red', zorder=5,
                                          label='Unstable Region (λ > μ)')

        ax.scatter(sample['lambda_per_min'], sample['mu_per_min'],
                  c=sample['Wait'], cmap='RdYlGn_r', s=20, alpha=0.4,
                  edgecolors='black', linewidth=0.3, label='Actual Data Points')

        ax.set_xlabel('λ (Arrival Rate, patients/min)', fontsize=13, fontweight='bold')
        ax.set_ylabel('μ (Service Rate, patients/min)', fontsize=13, fontweight='bold')
        ax.set_title('Queue Stability Phase Diagram\nOperations Research: System Capacity Analysis',
                    fontsize=16, fontweight='bold', pad=20)

        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')

        ax.text(0.02, 0.98, 'Stable Region\n(λ < μ)',
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

        ax.text(0.98, 0.02, 'Unstable Region\n(λ > μ)\nQueue grows unbounded',
               transform=ax.transAxes, fontsize=11, verticalalignment='bottom',
               horizontalalignment='right',
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Phase diagram saved to {save_path}")
        plt.close()
        return fig

    def create_phase_diagram_interactive(self, save_path='phase_diagram_interactive.html'):
        """
        Interactive version of phase diagram using Plotly
        """
        if 'lambda_per_min' not in self.df.columns or 'mu_per_min' not in self.df.columns:
            print("Error: lambda_per_min or mu_per_min not found")
            return None

        sample = self.df.sample(n=min(10000, len(self.df)), random_state=42)

        lambda_range = np.linspace(sample['lambda_per_min'].min(),
                                   sample['lambda_per_min'].max(), 100)
        mu_range = np.linspace(sample['mu_per_min'].min(),
                              sample['mu_per_min'].max(), 100)

        Lambda, Mu = np.meshgrid(lambda_range, mu_range)

        def theoretical_wait(lam, mu, servers=1):
            rho = lam / (servers * mu)
            if rho >= 1:
                return 500
            else:
                return (rho / (mu * (1 - rho)))

        Wait = np.vectorize(theoretical_wait)(Lambda, Mu, servers=sample['servers'].mode()[0] if 'servers' in sample.columns else 1)
        Wait = np.clip(Wait, 0, 500)

        fig = go.Figure()

        fig.add_trace(go.Contour(
            z=Wait,
            x=lambda_range,
            y=mu_range,
            colorscale='RdYlGn_r',
            colorbar=dict(title='Wait Time (min)'),
            contours=dict(
                start=0,
                end=500,
                size=25,
                showlabels=True
            ),
            hovertemplate='λ: %{x:.4f}<br>μ: %{y:.4f}<br>Wait: %{z:.1f} min<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x=lambda_range,
            y=lambda_range,
            mode='lines',
            line=dict(color='black', width=3, dash='dash'),
            name='λ = μ (Stability Boundary)',
            hoverinfo='skip'
        ))

        fig.add_trace(go.Scatter(
            x=sample['lambda_per_min'].sample(1000),
            y=sample['mu_per_min'].sample(1000),
            mode='markers',
            marker=dict(
                size=5,
                color=sample['Wait'].sample(1000),
                colorscale='RdYlGn_r',
                opacity=0.5,
                line=dict(width=0.5, color='black')
            ),
            name='Actual Data',
            hovertemplate='λ: %{x:.4f}<br>μ: %{y:.4f}<extra></extra>'
        ))

        fig.update_layout(
            title={
                'text': 'Queue Stability Phase Diagram<br><sub>Operations Research: System Capacity Analysis</sub>',
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title='λ (Arrival Rate, patients/min)',
            yaxis_title='μ (Service Rate, patients/min)',
            width=1000,
            height=800,
            font=dict(size=12),
            hovermode='closest'
        )

        fig.write_html(save_path)
        print(f"Interactive phase diagram saved to {save_path}")
        return fig

    def generate_all_visualizations(self, output_dir='visualizations'):
        """
        Generate all three core visualizations
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        print("Generating visualization suite for internship portfolio...")
        print("=" * 60)

        print("\n[1/6] Creating interactive heatmap...")
        self.create_heatmap_interactive(f'{output_dir}/wait_time_heatmap.html')

        print("\n[2/6] Creating static heatmap...")
        self.create_heatmap_static(f'{output_dir}/wait_time_heatmap_static.png')

        print("\n[3/6] Creating queue simulation animation...")
        self.create_queue_simulation_animation(f'{output_dir}/queue_simulation.gif')

        print("\n[4/6] Creating static phase diagram...")
        self.create_phase_diagram(f'{output_dir}/phase_diagram.png')

        print("\n[5/6] Creating interactive phase diagram...")
        self.create_phase_diagram_interactive(f'{output_dir}/phase_diagram_interactive.html')

        print("\n[6/6] Creating summary report...")
        self._create_summary_report(output_dir)

        print("\n" + "=" * 60)
        print(f"All visualizations generated successfully in '{output_dir}/' directory!")
        print("\nKey files:")
        print(f"  - wait_time_heatmap.html (Interactive)")
        print(f"  - queue_simulation.gif (Animation)")
        print(f"  - phase_diagram.png (Static)")
        print(f"  - phase_diagram_interactive.html (Interactive)")

    def _create_summary_report(self, output_dir):
        """Create a summary markdown report"""
        report = f"""# Wait-Time Analysis Visualization Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Summary
- Total Records: {len(self.df):,}
- Date Range: {self.df['arrival_time'].min()} to {self.df['arrival_time'].max()}
- Average Wait Time: {self.df['Wait'].mean():.2f} minutes
- Median Wait Time: {self.df['Wait'].median():.2f} minutes
- 90th Percentile Wait: {self.df['Wait'].quantile(0.9):.2f} minutes

## Visualizations Included

### 1. Interactive Wait-Time Heatmap (Hour × Day)
**File:** `wait_time_heatmap.html`

**What it shows:**
- When the clinic becomes congested
- Temporal rhythm of the system
- The "signature" of demand

**Why it's impressive:**
A 2-D heatmap shows emergent structure (like NYC subway traffic maps).
It's intuitive, beautiful, and tells a story in <5 seconds.
This is the kind of chart hospitals actually use for staffing decisions.

### 2. Live Queue Simulation Animation
**File:** `queue_simulation.gif`

**What it shows:**
- The dynamic evolution of the queue
- When λ (arrival rate) crosses μ (service rate)
- Bursts, bottlenecks, collapses in real time

**Why it's impressive:**
Internship reviewers immediately understand that you can model systems,
not just static data. Animations stand out.

### 3. Queue Stability Phase Diagram (λ vs μ)
**File:** `phase_diagram.png` and `phase_diagram_interactive.html`

**What it shows:**
- Regions where the system is stable vs unstable
- How much buffer capacity the clinic needs
- Visual proof that your model encodes queueing theory

**Why it's impressive:**
Most applicants cannot produce a classical operations-research visualization.
This one is instantly perceived as "PhD-level".

## Technical Notes

### Peak Congestion Times
"""

        peak_hours = self.df.groupby('hour')['Wait'].mean().nlargest(3)
        for hour, wait in peak_hours.items():
            report += f"- {hour:02d}:00 - Average wait: {wait:.1f} minutes\n"

        report += "\n### Peak Congestion Days\n"
        day_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        peak_days = self.df.groupby('day_of_week')['Wait'].mean().nlargest(3)
        for day, wait in peak_days.items():
            report += f"- {day_labels[day]} - Average wait: {wait:.1f} minutes\n"

        report += f"""
### System Stability Metrics
- Average λ (arrival rate): {self.df['lambda_per_min'].mean():.4f} patients/min
- Average μ (service rate): {self.df['mu_per_min'].mean():.4f} patients/min
- System utilization (ρ): {(self.df['lambda_per_min'] / self.df['mu_per_min']).mean():.2%}
- Unstable periods (λ > μ): {(self.df['lambda_per_min'] > self.df['mu_per_min']).sum() / len(self.df):.2%}

---
*Generated using advanced queueing theory and operations research principles*
"""

        with open(f'{output_dir}/REPORT.md', 'w') as f:
            f.write(report)

        print(f"Summary report saved to {output_dir}/REPORT.md")


if __name__ == "__main__":
    csv_path = r"C:\Users\mhdto\OneDrive\Documents\Project DWT\enriched_wait_data.csv"

    print("Loading data...")
    df = pd.read_csv(csv_path, parse_dates=['x_ArrivalDTTM', 'x_ScheduledDTTM', 'x_BeginDTTM'])

    print(f"Loaded {len(df):,} records")

    viz = WaitTimeVisualizations(df)

    viz.generate_all_visualizations()
