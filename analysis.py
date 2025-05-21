import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def load_data(csv_path: str) -> pd.DataFrame:
    """Load CSV data using pandas."""
    df = pd.read_csv(csv_path)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Drop missing values and convert numeric columns."""
    df = df.dropna()
    if 'Name' in df.columns:
        numeric_cols = df.columns.drop('Name')
    else:
        numeric_cols = df.columns
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    return df


def compute_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean, std, max, and min for numeric columns."""
    stats = df.describe().loc[['mean', 'std', 'max', 'min']]
    return stats


def create_plots(df: pd.DataFrame) -> plt.Figure:
    """Create score distribution plots for each subject."""
    subjects = [c for c in df.columns if c != 'Name']
    fig, axes = plt.subplots(len(subjects), 1, figsize=(8, 4 * len(subjects)))
    if len(subjects) == 1:
        axes = [axes]
    for ax, subj in zip(axes, subjects):
        ax.hist(df[subj], bins=10, color='skyblue', edgecolor='black')
        ax.set_title(f"Distribution of {subj} Scores")
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
    plt.tight_layout()
    return fig


def generate_summaries(df: pd.DataFrame) -> str:
    """Generate a text summary comparing each student's score to the mean."""
    summaries = []
    subjects = [c for c in df.columns if c != 'Name']
    means = df[subjects].mean()
    for _, row in df.iterrows():
        name = row.get('Name', 'Unknown')
        for subj in subjects:
            score = row[subj]
            mean_score = means[subj]
            if score > mean_score:
                comp = '높습니다'
            elif score < mean_score:
                comp = '낮습니다'
            else:
                comp = '같습니다'
            summaries.append(f"{name}의 {subj} 점수는 {score}점으로 평균보다 {comp}.")
    return "\n".join(summaries)


def create_pdf(fig: plt.Figure, summary: str, output_path: str) -> None:
    """Save plots and summary to a single PDF."""
    with PdfPages(output_path) as pdf:
        pdf.savefig(fig)
        plt.close(fig)
        from matplotlib import pyplot as _plt
        fig_text = _plt.figure(figsize=(8.27, 11.69))
        fig_text.text(0.05, 0.95, summary, wrap=True)
        fig_text.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        pdf.savefig(fig_text)
        _plt.close(fig_text)


def main() -> None:
    df = load_data('data.csv')
    df = preprocess(df)
    stats = compute_stats(df)
    print('Statistics:\n', stats)
    fig = create_plots(df)
    summary = generate_summaries(df)
    print('\nSummary:\n', summary)
    create_pdf(fig, summary, 'report.pdf')
    print('Report saved to report.pdf')


if __name__ == '__main__':
    main()
