import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.dpi'] = 100

_NUM_COLS = ['age', 'bmi', 'children', 'charges']
_CAT_COLS = ['sex', 'smoker', 'region']

def header(titile:str, pos:str='up'):
    if pos =='up':
        print("=" * 58)
        print(titile)
        print("=" * 58)
    else:
        print("\n" + "=" * 58)
        print(titile)
        print("=" * 58)
 
 

def eda_overview(df: pd.DataFrame) -> None:
    """Bagian 1 – info dataset, missing value, statistik deskriptif."""
    header("INFORMASI DATASET")
    print(f"Jumlah baris  : {df.shape[0]}")
    print(f"Jumlah kolom  : {df.shape[1]}")
    print(f"\nTipe data:")
    print(df.dtypes.to_string())

    header("MISSING VALUES","wkwkwk")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    print(pd.DataFrame({'Missing': missing, 'Persen (%)': missing_pct}).to_string())
    if missing.sum() == 0:
        print("\nTidak terdapat missing value pada dataset.")
    else:
        print(f"\nTotal missing value: {missing.sum()}")

    header("STATISTIK DESKRIPTIF — VARIABEL NUMERIK","hehe")
    print(df[_NUM_COLS].describe().T.round(2).to_string())

    header("DISTRIBUSI VARIABEL KATEGORIK","hehe")
    for col in _CAT_COLS:
        vc = df[col].value_counts()
        pct = (vc / len(df) * 100).round(1)
        print(f"\n{col.upper()}:")
        print(pd.DataFrame({'Count': vc, 'Persen (%)': pct}).to_string())


def eda_distribusi(df: pd.DataFrame) -> None:
    """Bagian 2 – histogram + KDE + boxplot numerik, bar chart kategorik."""
    colors_hist = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    fig.suptitle("Distribusi & Boxplot Variabel Numerik", fontsize=14, fontweight='bold')

    for i, (col, color) in enumerate(zip(_NUM_COLS, colors_hist)):
        sns.histplot(
            df[col], # type:ignore --- linter vscode saya terlalu strict (abaikan)
            kde=True, ax=axes[0, i], color=color,
                     edgecolor='white', alpha=0.8)
        axes[0, i].set_title(col.upper(), fontweight='bold')
        axes[0, i].set_xlabel(col)
        skew = df[col].skew()
        kurt = df[col].kurtosis()
        axes[0, i].text(0.97, 0.95, f'Skew: {skew:.2f}\nKurt: {kurt:.2f}',
                        transform=axes[0, i].transAxes, ha='right', va='top',
                        fontsize=8, bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

        sns.boxplot(y=df[col], ax=axes[1, i], color=color, width=0.5,
                    flierprops=dict(marker='o', markerfacecolor='red',
                                    markersize=4, alpha=0.5))
        axes[1, i].set_title(f'Boxplot {col.upper()}', fontweight='bold')

    plt.tight_layout()
    plt.show()

    print("Skewness & Kurtosis Variabel Numerik:")
    interp = [
        'Mendekati normal (sedikit right-skewed)',
        'Mendekati normal (sedikit right-skewed)',
        'Right-skewed (mayoritas 0–2 anak)',
        'Right-skewed kuat (klaim besar sedikit)',
    ]
    sk_df = pd.DataFrame({
        'Skewness': df[_NUM_COLS].skew().round(3),
        'Kurtosis': df[_NUM_COLS].kurtosis().round(3),
        'Interpretasi': interp,
    })
    print(sk_df.to_string())

    cat_info = {
        'sex':    (['#4C72B0', '#C44E52'], 'Jenis Kelamin'),
        'smoker': (['#55A868', '#C44E52'], 'Status Merokok'),
        'region': (['#4C72B0', '#55A868', '#C44E52', '#8172B2'], 'Wilayah'),
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Distribusi Variabel Kategorik", fontsize=14, fontweight='bold')

    for i, (col, (colors, title)) in enumerate(cat_info.items()):
        vc = df[col].value_counts()
        bars = axes[i].bar(vc.index, vc.values, color=colors[:len(vc)],
                           edgecolor='white', linewidth=1.2)
        axes[i].set_title(title, fontweight='bold', fontsize=12)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Jumlah')
        for bar, v in zip(bars, vc.values):
            axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                         f'{v}\n({v/len(df)*100:.1f}%)',
                         ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.show()


def eda_outlier(df: pd.DataFrame) -> None:
    """Bagian 3 – deteksi outlier metode IQR dengan tabel dan visualisasi."""
    header("DETEKSI OUTLIER — METODE IQR (Inter-Quartile Range)")

    outlier_summary = []
    for col in _NUM_COLS:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        outlier_summary.append({
            'Variabel': col, 'Q1': Q1, 'Q3': Q3, 'IQR': IQR,
            'Batas Bawah': lower, 'Batas Atas': upper,
            'n Outlier': n_out, 'Persen (%)': round(n_out / len(df) * 100, 2),
        })
        print(f"\n{col.upper()}")
        print(f"  Q1={Q1:.2f}, Q3={Q3:.2f}, IQR={IQR:.2f}")
        print(f"  Batas bawah : {lower:.2f} | Batas atas : {upper:.2f}")
        print(f"  Jumlah outlier : {n_out} ({n_out/len(df)*100:.1f}%)")

    colors_box = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("Deteksi Outlier per Variabel Numerik (IQR)",
                 fontsize=13, fontweight='bold')

    for i, (col, color) in enumerate(zip(_NUM_COLS, colors_box)):
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        is_out = (df[col] < lower) | (df[col] > upper)
        axes[i].scatter(np.zeros(is_out.sum()), df.loc[is_out, col],
                        color='red', s=20, zorder=5, alpha=0.7, label='Outlier')
        axes[i].boxplot(df[col], patch_artist=True,
                        boxprops=dict(facecolor=color, alpha=0.5),
                        medianprops=dict(color='black', linewidth=2))
        axes[i].set_title(f'{col.upper()}\n({is_out.sum()} outlier)', fontweight='bold')
        axes[i].set_xticks([])
        if i == 0:
            axes[i].legend(fontsize=8)

    plt.tight_layout()
    plt.show()

    print("\nRingkasan Outlier:")
    print(pd.DataFrame(outlier_summary)
            .set_index('Variabel')[['n Outlier', 'Persen (%)']].to_string())
    print("\nCatatan: Outlier pada 'charges' umumnya berasal dari perokok dengan BMI tinggi —")
    print("         bukan data error, melainkan kasus berisiko tinggi yang penting secara aktuaria.")


def eda_bivariat(df: pd.DataFrame) -> None:
    """Bagian 4 – analisis bivariat: charges vs setiap prediktor."""
    smoker_colors = df['smoker'].map({'yes': '#C44E52', 'no': '#4C72B0'})

    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    fig.suptitle("Analisis Bivariat: Charges vs Variabel Prediktor",
                 fontsize=14, fontweight='bold')

    # Scatter charges vs age
    r_age, _ = stats.pearsonr(df['age'], df['charges'])
    axes[0, 0].scatter(df['age'], df['charges'], c=smoker_colors, alpha=0.4, s=18)
    m, b = np.polyfit(df['age'], df['charges'], 1)
    x_line = np.linspace(df['age'].min(), df['age'].max(), 200)
    axes[0, 0].plot(x_line, m * x_line + b, color='black', linewidth=1.5, linestyle='--')
    axes[0, 0].set_xlabel('Age'); axes[0, 0].set_ylabel('Charges')
    axes[0, 0].set_title(f'Charges vs Age  |  r = {r_age:.3f}', fontweight='bold')
    axes[0, 0].text(0.05, 0.93, 'Merah = Smoker\nBiru = Non-smoker',
                    transform=axes[0, 0].transAxes, fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7))

    # Scatter charges vs bmi
    r_bmi, _ = stats.pearsonr(df['bmi'], df['charges'])
    axes[0, 1].scatter(df['bmi'], df['charges'], c=smoker_colors, alpha=0.4, s=18)
    m2, b2 = np.polyfit(df['bmi'], df['charges'], 1)
    x_line2 = np.linspace(df['bmi'].min(), df['bmi'].max(), 200)
    axes[0, 1].plot(x_line2, m2 * x_line2 + b2, color='black', linewidth=1.5, linestyle='--')
    axes[0, 1].set_xlabel('BMI'); axes[0, 1].set_ylabel('Charges')
    axes[0, 1].set_title(f'Charges vs BMI  |  r = {r_bmi:.3f}', fontweight='bold')
    axes[0, 1].axvline(30, color='orange', linestyle=':', linewidth=1.5, label='BMI=30 (Obese)')
    axes[0, 1].legend(fontsize=8)

    # Boxplot charges vs smoker
    sns.boxplot(x='smoker', y='charges', data=df, ax=axes[0, 2],
                palette={'yes': '#C44E52', 'no': '#4C72B0'}, order=['no', 'yes'])
    medians = df.groupby('smoker')['charges'].median()
    for j, cat in enumerate(['no', 'yes']):
        axes[0, 2].text(j, medians[cat] + 500, f'Median:\n{medians[cat]:,.0f}',
                        ha='center', fontsize=8, fontweight='bold')
    axes[0, 2].set_title('Charges vs Smoker', fontweight='bold')
    axes[0, 2].set_xlabel('Smoker'); axes[0, 2].set_ylabel('Charges')

    # Boxplot charges vs region
    order_region = df.groupby('region')['charges'].median().sort_values().index
    sns.boxplot(x='region', y='charges', data=df, ax=axes[1, 0],
                order=order_region, palette='Set2')
    axes[1, 0].set_title('Charges vs Region', fontweight='bold')
    axes[1, 0].tick_params(axis='x', rotation=15)
    axes[1, 0].set_xlabel('Region'); axes[1, 0].set_ylabel('Charges')

    # Boxplot charges vs sex
    sns.boxplot(x='sex', y='charges', data=df, ax=axes[1, 1],
                palette={'male': '#4C72B0', 'female': '#C44E52'})
    axes[1, 1].set_title('Charges vs Sex', fontweight='bold')
    axes[1, 1].set_xlabel('Sex'); axes[1, 1].set_ylabel('Charges')

    # Boxplot charges vs children
    sns.boxplot(x='children', y='charges', data=df, ax=axes[1, 2], palette='Blues')
    axes[1, 2].set_title('Charges vs Children', fontweight='bold')
    axes[1, 2].set_xlabel('Jumlah Anak'); axes[1, 2].set_ylabel('Charges')

    plt.tight_layout()
    plt.show()

    print("Median & Mean Charges per Kelompok:")
    for col in ['smoker', 'sex', 'region', 'children']:
        print(f"\n{col.upper()}:")
        print(df.groupby(col)['charges']
                .agg(Median='median', Mean='mean', Std='std')
                .round(2).to_string())


def eda_korelasi(df: pd.DataFrame) -> None:
    """Bagian 5 – correlation heatmap, bar chart, dan pairplot."""
    df_enc = df.copy()
    df_enc['sex_enc']    = df['sex'].map({'male': 0, 'female': 1})
    df_enc['smoker_enc'] = df['smoker'].map({'no': 0, 'yes': 1})
    df_enc['region_enc'] = df['region'].astype('category').cat.codes

    cols_corr   = ['age', 'bmi', 'children', 'sex_enc', 'smoker_enc', 'region_enc', 'charges']
    labels_corr = ['Age', 'BMI', 'Children', 'Sex', 'Smoker', 'Region', 'Charges']
    corr_mat    = df_enc[cols_corr].corr()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Analisis Korelasi Antar Variabel", fontsize=14, fontweight='bold')

    mask = np.zeros_like(corr_mat, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    sns.heatmap(corr_mat, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                xticklabels=labels_corr, yticklabels=labels_corr, ax=axes[0],
                square=True, linewidths=0.5, vmin=-1, vmax=1, annot_kws={'size': 9},
                mask=mask)
    axes[0].set_title('Correlation Heatmap (semua variabel)', fontweight='bold')

    corr_charges = corr_mat['charges'].drop('charges')
    corr_abs     = corr_charges.abs().sort_values()
    labels_bar   = [labels_corr[cols_corr.index(c)] for c in corr_abs.index]
    bar_colors   = ['#C44E52' if v > 0.5 else '#DD8452' if v > 0.2 else '#4C72B0'
                    for v in corr_abs.values]

    bars = axes[1].barh(labels_bar, corr_abs.values, color=bar_colors, edgecolor='white')
    axes[1].set_title('Korelasi Absolut terhadap Charges\n(merah > 0.5 | oranye > 0.2)',
                      fontweight='bold')
    axes[1].set_xlabel('|Korelasi Pearson|')
    axes[1].axvline(0.5, color='red',    linestyle='--', alpha=0.5, linewidth=1.2)
    axes[1].axvline(0.2, color='orange', linestyle='--', alpha=0.5, linewidth=1.2)
    axes[1].set_xlim(0, 1)
    for bar, v in zip(bars, corr_abs.values):
        axes[1].text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                     f'{v:.3f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.show()

    print("Korelasi terhadap Charges (diurutkan secara absolut):")
    corr_tbl = pd.DataFrame({
        'Korelasi':    corr_charges.round(4),
        '|Korelasi|':  corr_charges.abs().round(4),
        'Kekuatan':    corr_charges.abs().map(
            lambda v: 'Kuat' if v > 0.5 else 'Sedang' if v > 0.2 else 'Lemah'
        ),
    }).loc[corr_abs.sort_values(ascending=False).index]
    corr_tbl.index = labels_bar[::-1]
    print(corr_tbl.to_string())

    print("\nMembuat pairplot (harap tunggu)...")
    g = sns.pairplot(
        df, vars=['age', 'bmi', 'charges'],
        hue='smoker', palette={'yes': '#C44E52', 'no': '#4C72B0'},
        plot_kws={'alpha': 0.4, 's': 15},
        diag_kind='kde',
        height=2.8,
    )
    g.figure.suptitle("Pairplot: Age, BMI, Charges — diwarnai status merokok",
                   y=1.02, fontsize=12, fontweight='bold')
    plt.show()
