import sys
import preprocessing as pp

if __name__ == '__main__':

    if sys.argv[1] == 'merge':
        pp.merge_text_files(pp.RAW_DIR, pp.join(pp.PREPROCESSED_DIR, 'raw_merged.tsv'))
    elif sys.argv[1] == 'to_sqlite':
        pp.raw_tsv_to_sqlite()
    elif sys.argv[1] == 'tsv_vs_sqlite':
        pp.tsv_vs_sqlite()
    elif sys.argv[1] == 'group_cc':
        pp.group_by_country_codes()
    elif sys.argv[1] == 'merge_to_size':
        pp.merge_to_size(float(sys.argv[2]), pp.GROUPED_CC_DIR, pp.MERGED_TO_SIZE_DIR)
