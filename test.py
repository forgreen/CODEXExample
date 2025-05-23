from split_data_module import SplitDataModule, read_csv


def show_split(label, module, data):
    try:
        first, second = module.split(data)
        print(f"{label} TRAIN", first)
        print(f"{label} TEST", second)
    except Exception as e:
        print(f"{label} ERROR:", e)


def main():
    data = read_csv('data.csv')

    # Default row-based split (60/40 by default)
    mod_rows = SplitDataModule(splitting_mode='rows', fraction=0.6, random_seed=42)
    show_split('ROWS', mod_rows, data)

    # Expression-based split (Math > 80)
    mod_expr = SplitDataModule(splitting_mode='expression', expression='Math > 80')
    show_split('EXPRESSION', mod_expr, data)

    # Regex split (Name begins with 'A')
    mod_regex = SplitDataModule(splitting_mode='regex', regex='^A', target_column='Name')
    show_split('REGEX', mod_regex, data)

    # Recommender split (will fail due to dataset format)
    mod_recomm = SplitDataModule(splitting_mode='recommender', fraction=0.5)
    show_split('RECOMMENDER', mod_recomm, data)

    # Stratified row split using Science column as key
    mod_strat = SplitDataModule(
        splitting_mode='rows',
        fraction=0.5,
        stratified=True,
        stratify_key='Science'
    )
    show_split('STRATIFIED', mod_strat, data)


if __name__ == '__main__':
    main()
