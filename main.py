#------------------------ Header Comment ----------------------#
#The main program of the entire project

from src.load_data import load_battery_data
from src.visualize import plot_capacity_curve, plot_soh_curve, plot_eol_hist, plot_per_battery
from src.train_model import train_and_evaluate

#Run the full pipeline: data → plots → training
def main():
    print("🔋 Battery RUL Final - Polished Pipeline")
    df = load_battery_data('data')
    if df is None or df.empty:
        print("⚠️ No usable data. Please place .mat files into the data/ folder.")
        return
    plot_capacity_curve(df, save_path='outputs/capacity_curve.png')
    plot_soh_curve(df, save_path='outputs/soh_curve.png')
    plot_eol_hist(df, save_path='outputs/eol_cycle_hist.png')
    plot_per_battery(df, out_dir='outputs/plots_per_file')
    train_and_evaluate(df, output_dir='outputs')

if __name__ == "__main__":
    main()
