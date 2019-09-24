import csv
import json
import logging
from random import sample

# from sklearn.cluster import DBSCAN
import h5py
import matplotlib.pyplot as plt
import numpy as np
from allennlp.modules.elmo import Elmo
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from wordcloud import WordCloud

logging.basicConfig(level=logging.INFO)

options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

elmo = Elmo(options_file, weight_file, 2, dropout=0)


def translate_vectors_to_9d(vectors):
    vectors_9d = []
    for vector in vectors:
        vectors_9d.append([translate_16d_to_9d(vector[0])])
    return vectors_9d


def translate_16d_to_9d(vector_16d):
    vector_9d = []
    is_it_ok = False
    for i in range(0, 8):
        if vector_16d[i] > 0:
            vector_9d.append(i + 1)
            is_it_ok = True
            break
    if not is_it_ok:
        vector_9d.append(0)
    vector_9d.extend(vector_16d[8:])
    return vector_9d


def write_2d_scatter_comparison_plot(dict_data, address, dim):
    # how to create a scatterplot
    arr = np.empty((0, dim), dtype='f')

    vectors_data = []
    color_data = []
    legend_data = []

    for key, dict_d in dict_data.items():
        vectors_data.append(dict_d['dataset'])
        color_data.append(dict_d['color'])
        legend_data.append(key)

    idxs = []
    val = 0
    idxs.append(0)
    for v in vectors_data:
        for vv in v:
            wrd_vector = vv[0]
            arr = np.append(arr, np.array([wrd_vector]), axis=0)
        val += len(v)
        idxs.append(val)

    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]

    idx = 0
    for c in color_data:
        plt.scatter(x_coords[idxs[idx]:idxs[idx + 1]], y_coords[idxs[idx]:idxs[idx + 1]], s=5, c=c)
        idx += 1

    plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
    plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)

    plt.savefig(address, format="png", figsize=(2000, 1200), dpi=300)
    plt.clf()
    plt.cla()
    plt.close()


def write_ngrams_as_csv(list_encding_strings, max_length, address):
    lines = []
    lines.append(['pattern', 'k', 'frequency', 'expected occurrences', 'percentage'])
    all_dict = {}
    for encoding_string in list_encding_strings:
        anaylzed_dict = generate_ngrams_of_codes(encoding_string, max_length)
        all_dict = merge_n_gram_dicts(all_dict, anaylzed_dict)

    for pattern, statistics_dict in all_dict.items():
        line = []
        line.append(pattern)
        line.append(len(pattern))
        line.append(statistics_dict['frequency'])
        line.append(statistics_dict['frequency'] / len(list_encding_strings))
        line.append(statistics_dict['occurred'] / len(list_encding_strings) * 100)
        lines.append(line)

    with open(address, 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(lines)


def generate_ngrams_of_codes(encoding_string, max_length):
    res_dict = dict()
    for k in range(1, max_length):
        if k < len(encoding_string):
            k_grams = [encoding_string[i:i + k] for i in range(len(encoding_string) - k + 1)]
            for k_gram in k_grams:
                if k_gram not in res_dict.keys():
                    res_dict[k_gram] = {'frequency': 0, 'occurred': 1}
                res_dict[k_gram]['frequency'] = res_dict[k_gram]['frequency'] + 1
    return res_dict


def merge_n_gram_dicts(all_dict, analyzed_dict):
    new_dict = all_dict
    for k, v in analyzed_dict.items():
        if k in new_dict.keys():
            new_dict[k]['frequency'] = new_dict[k]['frequency'] + analyzed_dict[k]['frequency']
            new_dict[k]['occurred'] = new_dict[k]['occurred'] + 1
        else:
            new_dict[k] = analyzed_dict[k]
    return new_dict


def merge_dicts(d1, d2):
    new_dict = d1
    for k, v in d2.items():
        new_dict[k].extend(v)
    return new_dict


def write_2d_scatter_plot(list_embedding, list_words, address, dim):
    arr = np.empty((0, dim), dtype='f')

    for wrd_score in list_embedding:
        wrd_vector = wrd_score[0]
        arr = np.append(arr, np.array([wrd_vector]), axis=0)

    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]

    # display scatter plot
    plt.scatter(x_coords, y_coords, s=3)

    for label, x, y in zip(list_words, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', fontsize=6)
    plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
    plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)

    plt.axis("off")
    plt.savefig(address, format="png", figsize=(2000, 1200), dpi=300)
    plt.clf()
    plt.cla()
    plt.close()


def write_csv_frequency(address, info_dict):
    lines = []
    for k, v in info_dict.items():
        line = []
        line.append(k)
        line.append(v)
        lines.append(line)
    with open(address, 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(lines)


def read_msmarco_code(fname):
    res = []
    with open(fname) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')

        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            else:
                if row is not None and len(row) > 0:
                    res.append(row[0])
                    line_count += 1
        return res


def generate_vectors_ms_marco(codes):
    dict_output = {'question_ids': [], 'sources': [], 'codes': [], 'vectors': [], 'cluster_ids': []}
    dimensions = ['1', '2', '3', '4', '5', '6', '7', '8', 'n', 't', 'o', 'q', 'p', 's', 'a', 'r']
    vectors = []
    count = 1
    for code in codes:
        dict_output['question_ids'].append(count)
        count += 1
        dict_output['sources'].append('MS MARCO')
        dict_output['codes'].append(code)
        vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for d in dimensions:
            if d in code:
                count = code.count(d)
                idx = dimensions.index(d)
                vector[idx] = count
        vectors.append([vector])
    dict_output['vectors'] = vectors
    return dict_output


def get_random_sample(list, size):
    return sample(list, size)


def write_word_clouds(address, bulk_text):
    wordcloud = WordCloud(width=800, height=600, max_font_size=150, max_words=100, background_color="white").generate(
        bulk_text)
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(address, format="png", figsize=(2000, 1400), dpi=300)
    plt.clf()
    plt.cla()
    plt.close()


def read_file(address):
    if address:
        with open(address, 'r') as f:
            return json.load(f)


def generate_based_on_category(address):
    dict_subdatasets = {}
    colors = ['red', 'yellow', 'green', 'blue', 'purple', 'orange', 'black', 'brown']
    dimensions = ['1', '2', '3', '4', '5', '6', '7', '8', 'n', 't', 'o', 'q', 'p', 's', 'a', 'r']
    results = read_file(address)
    vectors = []
    sources = []
    q_types = []
    sources_dicts = {}  # dict_datasets2['GeoAnalytical'] = {'dataset': translate_vectors_to_9d(d2), 'color': 'blue'}
    q_types_dicts = {}
    for result in results:
        all_info = result['all_info']
        vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in all_info:
            if i['tag'] != '' and i['tag'] in dimensions:
                idx = dimensions.index(i['tag'])
                vector[idx] = vector[idx] + 1
        vectors.append([vector])
        source = result['source']
        sources.append(source)
        if source not in sources_dicts:
            color = colors[len(sources_dicts.keys())]
            logging.info('color {} is assigned to the source {}'.format(color, source))
            sources_dicts[source] = {'dataset': [], 'color': color}
        sources_dicts[source]['dataset'].append([vector])
        q_type = result['q_type']
        q_types.append(q_type)
        if q_type not in q_types_dicts:
            color = colors[len(q_types_dicts.keys())]
            logging.info('color {} is assigned to q_type {}'.format(color, q_type))
            q_types_dicts[q_type] = {'dataset': [], 'color': color}
        q_types_dicts[q_type]['dataset'].append([vector])

    write_2d_scatter_comparison_plot(sources_dicts, 'graphs/2d_encoding_comparison_source.png', 16)
    write_2d_scatter_comparison_plot(q_types_dicts, 'graphs/2d_encoding_comparison_q_type.png', 16)


def analyze_and_visualize(address, dataset_name):
    wh_encodings = ['1', '2', '3', '4', '5', '6', '7', '8']
    dimensions = ['1', '2', '3', '4', '5', '6', '7', '8', 'n', 't', 'o', 'q', 'p', 's', 'a', 'r']

    results = read_file(address)

    dict_output = {'question_ids': [], 'sources': [], 'codes': [], 'vectors': [], 'cluster_ids': []}

    list_encoding_strings = []

    questions = ''
    pnames = ''
    ptypes = ''
    objects = ''
    intents = ''
    ointents = ''
    tintents = ''
    situations = ''
    activities = ''
    qualities = ''
    oqualities = ''

    pnset = set()
    tset = set()
    oset = set()
    sset = set()
    aset = set()
    qset = set()
    oqset = set()
    intent_set = set()

    pn_dict = dict()
    t_dict = dict()
    o_dict = dict()
    s_dict = dict()
    a_dict = dict()
    q_dict = dict()
    r_dict = dict()
    oq_dict = dict()
    ti_dict = dict()
    oi_dict = dict()
    qi_dict = dict()
    i_dict = dict()

    vectors = []
    for result in results:
        dict_output['sources'].append(dataset_name)
        dict_output['question_ids'].append(result['id'])
        dict_output['codes'].append(result['all_code'])
        list_encoding_strings.append(result['all_code'])
        questions += result['question'] + ' '
        all_info = result['all_info']
        vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in all_info:
            if i['tag'] == 'n':
                pnames += i['value'] + ' '
                pnset.add(i['value'].strip())
                if i['value'].lower().strip() not in pn_dict.keys():
                    pn_dict[i['value'].lower().strip()] = 0
                pn_dict[i['value'].lower().strip()] = pn_dict[i['value'].lower().strip()] + 1
            elif i['tag'] == 't':
                ptypes += i['value'] + ' '
                tset.add(i['value'].strip())
                if i['value'].lower().strip() not in t_dict.keys():
                    t_dict[i['value'].lower().strip()] = 0
                t_dict[i['value'].lower().strip()] = t_dict[i['value'].lower().strip()] + 1
            elif i['tag'] == 'o':
                objects += i['value'] + ' '
                oset.add(i['value'].strip())
                if i['value'].lower().strip() not in o_dict.keys():
                    o_dict[i['value'].lower().strip()] = 0
                o_dict[i['value'].lower().strip()] = o_dict[i['value'].lower().strip()] + 1
            elif i['tag'] == 's':
                situations += i['value'] + ' '
                sset.add(i['value'].strip())
                if i['value'].lower().strip() not in s_dict.keys():
                    s_dict[i['value'].lower().strip()] = 0
                s_dict[i['value'].lower().strip()] = s_dict[i['value'].lower().strip()] + 1
            elif i['tag'] == 'a':
                activities += i['value'] + ' '
                aset.add(i['value'].strip())
                if i['value'].lower().strip() not in a_dict.keys():
                    a_dict[i['value'].lower().strip()] = 0
                a_dict[i['value'].lower().strip()] = a_dict[i['value'].lower().strip()] + 1
            elif i['tag'] == 'q':
                qualities += i['value'] + ' '
                qset.add(i['value'].strip())
                if i['value'].lower().strip() not in q_dict.keys():
                    q_dict[i['value'].lower().strip()] = 0
                q_dict[i['value'].lower().strip()] = q_dict[i['value'].lower().strip()] + 1
            elif i['tag'] == 'p':
                oqualities += i['value'] + ' '
                oqset.add(i['value'].strip())
                if i['value'].lower().strip() not in oq_dict.keys():
                    oq_dict[i['value'].lower().strip()] = 0
                oq_dict[i['value'].lower().strip()] = oq_dict[i['value'].lower().strip()] + 1
            elif i['tag'] == 'r':
                if i['value'].lower().strip() not in r_dict.keys():
                    r_dict[i['value'].lower().strip()] = 0
                r_dict[i['value'].lower().strip()] = r_dict[i['value'].lower().strip()] + 1

            if i['tag'] != '' and i['tag'] in dimensions:
                idx = dimensions.index(i['tag'])
                vector[idx] = vector[idx] + 1
        vectors.append([vector])

        intent_info = result['intent_info']
        for i in intent_info:
            if i['tag'] == 'o':
                ointents += i['value'] + ' '
                intent_set.add(i['value'])
                if i['value'].lower().strip() not in oi_dict.keys():
                    oi_dict[i['value'].lower().strip()] = 0
                oi_dict[i['value'].lower().strip()] = oi_dict[i['value'].lower().strip()] + 1
            elif i['tag'] == 't':
                tintents += i['value'] + ' '
                intent_set.add(i['value'])
                if i['value'].lower().strip() not in ti_dict.keys():
                    ti_dict[i['value'].lower().strip()] = 0
                ti_dict[i['value'].lower().strip()] = ti_dict[i['value'].lower().strip()] + 1
            elif i['tag'] in wh_encodings:
                if i['value'].lower().strip() not in i_dict.keys():
                    i_dict[i['value'].lower().strip()] = 0
                i_dict[i['value'].lower().strip()] = i_dict[i['value'].lower().strip()] + 1

    # generate word clouds
    write_word_clouds('graphs/{}/pnames.png'.format(dataset_name), pnames)
    write_word_clouds('graphs/{}/ptypes.png'.format(dataset_name), ptypes)
    write_word_clouds('graphs/{}/objects.png'.format(dataset_name), objects)
    write_word_clouds('graphs/{}/situations.png'.format(dataset_name), situations)
    write_word_clouds('graphs/{}/activities.png'.format(dataset_name), activities)
    write_word_clouds('graphs/{}/qualities.png'.format(dataset_name), qualities)
    write_word_clouds('graphs/{}/oqualities.png'.format(dataset_name), oqualities)
    write_word_clouds('graphs/{}/questions.png'.format(dataset_name), questions)
    write_word_clouds('graphs/{}/ointents.png'.format(dataset_name), ointents)
    write_word_clouds('graphs/{}/tintents.png'.format(dataset_name), tintents)

    write_csv_frequency('csv/{}/pnames.csv'.format(dataset_name), pn_dict)
    write_csv_frequency('csv/{}/ptypes.csv'.format(dataset_name), t_dict)
    write_csv_frequency('csv/{}/objects.csv'.format(dataset_name), o_dict)
    write_csv_frequency('csv/{}/situations.csv'.format(dataset_name), s_dict)
    write_csv_frequency('csv/{}/activities.csv'.format(dataset_name), a_dict)
    write_csv_frequency('csv/{}/qualities.csv'.format(dataset_name), q_dict)
    write_csv_frequency('csv/{}/oqualities.csv'.format(dataset_name), oq_dict)
    write_csv_frequency('csv/{}/ointents.csv'.format(dataset_name), oi_dict)
    write_csv_frequency('csv/{}/tintents.csv'.format(dataset_name), ti_dict)
    write_csv_frequency('csv/{}/relations.csv'.format(dataset_name), r_dict)
    write_csv_frequency('csv/{}/intents.csv'.format(dataset_name), i_dict)

    # generate embeddings
    # pnset = list(pnset)
    # pnemb = elmo(batch_to_ids([[v] for v in pnset]))['elmo_representations'][0].detach().numpy()
    # tset = list(tset)
    # temb = elmo(batch_to_ids([[v] for v in tset]))['elmo_representations'][0].detach().numpy()
    # oset = list(oset)
    # oemb = elmo(batch_to_ids([[v] for v in oset]))['elmo_representations'][0].detach().numpy()
    # sset = list(sset)
    # semb = elmo(batch_to_ids([[v] for v in sset]))['elmo_representations'][0].detach().numpy()
    # aset = list(aset)
    # aemb = elmo(batch_to_ids([[v] for v in aset]))['elmo_representations'][0].detach().numpy()
    # qset = list(qset)
    # qemb = elmo(batch_to_ids([[v] for v in qset]))['elmo_representations'][0].detach().numpy()
    # oqset = list(oqset)
    # oqemb = elmo(batch_to_ids([[v] for v in oqset]))['elmo_representations'][0].detach().numpy()
    # intent_set = list(intent_set)
    # intentemb = elmo(batch_to_ids([[v] for v in intent_set]))['elmo_representations'][0].detach().numpy()
    #
    # # generate embeddings plot
    # write_2d_scatter_plot(temb, tset, 'graphs/{}/1-type-emb-sp.png'.format(dataset_name), 1024)
    # write_2d_scatter_plot(pnemb, pnset, 'graphs/{}/1-pnames-emb-sp.png'.format(dataset_name), 1024)
    # write_2d_scatter_plot(oemb, oset, 'graphs/{}/1-object-emb-sp.png'.format(dataset_name), 1024)
    # write_2d_scatter_plot(semb, sset, 'graphs/{}/1-situation-emb-sp.png'.format(dataset_name), 1024)
    # write_2d_scatter_plot(aemb, aset, 'graphs/{}/1-activity-emb-sp.png'.format(dataset_name), 1024)
    # write_2d_scatter_plot(qemb, qset, 'graphs/{}/1-quality-emb-sp.png'.format(dataset_name), 1024)
    # write_2d_scatter_plot(oqemb, oqset, 'graphs/{}/1-oquality-emb-sp.png'.format(dataset_name), 1024)
    # write_2d_scatter_plot(intentemb, intent_set, 'graphs/{}/1-intent-emb-sp.png'.format(dataset_name), 1024)

    write_ngrams_as_csv(list_encoding_strings, 5, 'ngrams/{}-ngrams.csv'.format(dataset_name))
    dict_output['vectors'] = vectors
    return dict_output


def encoding_cluster(datasets_dictionary):
    # ds_dicts = {'question_ids': [], 'sources': [], 'codes': [], 'vectors': [], 'cluster_ids' : []}
    logging.info('encoding-based clustering started...')
    vectors = datasets_dictionary['vectors']
    arr = np.empty((0, 16), dtype='f')
    idxs = []
    val = 0
    idxs.append(0)
    for v in vectors:
        for vv in v:
            wrd_vector = vv
            arr = np.append(arr, np.array([wrd_vector]), axis=0)
        val += len(v)
        idxs.append(val)
    X = arr

    # n_clusters = 4
    # logging.info('number of clusters are set to {}'.format(n_clusters))
    db = AgglomerativeClustering().fit(X)
    # db = Birch(n_clusters=n_clusters).fit(arr)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    # core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    datasets_dictionary['cluster_ids'] = labels
    logging.info('clustering is finished...')
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    logging.info('number of noise points: {}'.format(n_noise_))
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    colors_dict = {}
    for k, col in zip(unique_labels, colors):
        colors_dict[k] = col
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        tsne = TSNE(n_components=2, random_state=0)
        np.set_printoptions(suppress=True)
        Y = tsne.fit_transform(X)
        logging.info('encoding representation\'s dimension is reduced to 2')

        xy = Y[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = Y[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('number of clusters: {}'.format(n_clusters_))
    # plt.show()
    address = 'graphs/clustering_of_encodings.png'
    logging.info('saving clustering plot in {}'.format(address))
    plt.savefig(address, format="png", figsize=(2000, 1200), dpi=300)
    plt.clf()
    plt.cla()
    plt.close()

    logging.info('calculating the percentage of questions of each source in each cluster')
    label_count = dict()
    for label in labels:
        if label not in label_count.keys():
            label_count[label] = 0
        label_count[label] = label_count[label] + 1
    source_count = dict()
    sources = set(datasets_dictionary['sources'])
    for source in sources:
        if source not in source_count:
            source_count[source] = 0
    for source in datasets_dictionary['sources']:
        source_count[source] = source_count[source] + 1

    label_source_count = {}
    for i in range(0, len(labels)):
        label_source_key = datasets_dictionary['sources'][i] + '--' + str(labels[i])
        if label_source_key not in label_source_count.keys():
            label_source_count[label_source_key] = 0
        label_source_count[label_source_key] = label_source_count[label_source_key] + 1
    source_precentages = []
    label_precentages = []
    for i in range(0, len(labels)):
        label_source_key = datasets_dictionary['sources'][i] + '--' + str(labels[i])
        l_s_count = label_source_count[label_source_key]
        label_precentages.append(l_s_count / label_count[labels[i]] * 100)
        source_precentages.append(l_s_count / source_count[datasets_dictionary['sources'][i]] * 100)
    datasets_dictionary['source_percentages'] = source_precentages
    datasets_dictionary['label_percentages'] = label_precentages

    result_address = 'data/geoaq/clustering.csv'
    datasets_dictionary['vectors'] = []
    logging.info('writing the results into {}'.format(result_address))
    lines = []
    lines.append(['source', 'question id', 'encoding', 'cluster id', 'source percentage', 'label percentage', 'color'])
    for i in range(0, len(labels)):
        # {'question_ids': [], 'sources': [], 'codes': [], 'vectors': [], 'cluster_ids': []}
        source = datasets_dictionary['sources'][i]
        question_id = datasets_dictionary['question_ids'][i]
        code = datasets_dictionary['codes'][i]
        cluster_id = datasets_dictionary['cluster_ids'][i]
        source_precentage = datasets_dictionary['source_percentages'][i]
        label_precentage = datasets_dictionary['label_percentages'][i]
        color = colors_dict[labels[i]]
        line = [source, question_id, code, cluster_id, source_precentage, label_precentage, color]
        lines.append(line)

    with open(result_address, 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(lines)


def load_raw_questions(fraw):  # g geoanalytical, geoqa: q, msmarco: m
    f = open(fraw, 'r')
    # M is the map from sentence to id
    M = dict()
    count = 0
    for line in f.readlines():
        line = line.replace('\n', '')
        M[str(count)] = [line]
        count += 1
    return M


def load_hdf_data(frep, M):
    f = h5py.File(frep, 'r')
    X = []
    S = []
    for key in f.keys():
        try:
            rep = f[key].value
            rep = np.mean(rep, axis=0)
            for i in range(len(M[key])):
                X.append(rep)
                S.append([key, M[key][i]])
        except:
            continue
    X = np.array(X)
    print(X.shape)

    return S, X


def emb_clustering(path_address):
    address_hdf5 = path_address + '/hdf5/'
    address_raw = path_address + '/data/'

    address_raw_simon_ds = address_raw + 'geoanalytical.txt'
    raw_simon_ds = load_raw_questions(address_raw_simon_ds)
    address_raw_201_ds = address_raw + 'geoqa.txt'
    raw_201_ds = load_raw_questions(address_raw_201_ds)
    address_raw_msmarco = address_raw + 'msmarco.txt'
    raw_msmarco_ds = load_raw_questions(address_raw_msmarco)

    address_hdf5_simon_ds = address_hdf5 + 'geoanalytical.hdf5'
    hdf5_simon_ds_S, hdf5_simon_ds_X = load_hdf_data(address_hdf5_simon_ds, raw_simon_ds)
    address_hdf5_201_ds = address_hdf5 + 'geoqa.hdf5'
    hdf5_201_ds_S, hdf5_201_ds_X = load_hdf_data(address_hdf5_201_ds, raw_201_ds)
    address_hdf5_msmarco = address_hdf5 + 'msmarco.hdf5'
    hdf5_msmarco_S, hdf5_msmarco_X = load_hdf_data(address_hdf5_msmarco, raw_msmarco_ds)
    logging.info('all data are loaded for embedding clustering')

    X = np.empty((0, 1024), dtype='f')
    X = np.append(X, hdf5_simon_ds_X, axis=0)
    X = np.append(X, hdf5_201_ds_X, axis=0)
    X = np.append(X, hdf5_msmarco_X, axis=0)

    S = hdf5_simon_ds_S
    S.extend(hdf5_201_ds_S)
    S.extend(hdf5_msmarco_S)
    logging.info('merging all data, S size: {}'.format(len(S)))
    logging.info('merging all data, X shape: {}'.format(X.shape))
    logging.info('wait here...')

    db = AgglomerativeClustering().fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    labels = db.labels_

    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(X)
    logging.info('embedding representation\'s dimension is reduced to 2')

    unique_labels = set(labels)
    n_clusters = len(unique_labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    colors_dict = {}
    for k, col in zip(unique_labels, colors):
        colors_dict[k] = col
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = Y[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = Y[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('number of clusters: {}'.format(n_clusters))
    # plt.show()
    address = 'graphs/clustering_of_embeddings.png'
    logging.info('saving clustering plot in {}'.format(address))
    plt.savefig(address, format="png", figsize=(2000, 1200), dpi=300)
    plt.clf()
    plt.cla()
    plt.close()

    counter = 0
    lines = []
    lines.append(['s_id', 'source', 'question', 'cluster id'])
    for label in labels:
        source = ''
        if counter < 429:
            source = 'Simon dataset'
        elif counter < 629 and counter >= 429:
            source = 'GeoQA 201 dataset'
        else:
            source = 'MS MARCO dataset'
        question = S[counter][1]
        line = [counter, source, question, label]
        lines.append(line)
        counter += 1
    with open('data/clustering_embedding.csv', 'w') as writeFile:
        writer = csv.writer(writeFile)
        writer.writerows(lines)


logging.info('analyzing 201 GS GeoQA dataset...')
d1 = analyze_and_visualize('data/geoaq/201-GSQuestions-dataset/analyzed_question_new_parser.json',
                           '201-GSGeoQA-Dataset')
logging.info('analyzing GeoAnalytical questions dataset...')
d2 = analyze_and_visualize('data/geoaq/Simon-dataset/analyzed_question_new_parser.json', 'Simon-dataset')
logging.info('analyzing MS MARCO (Unique) dataset...')
# ms_marco_unique = read_msmarco_code('data/geoaq/unique_codes_msmarco.csv')
# d3 = generate_vectors_ms_marco(ms_marco_unique)
d3 = analyze_and_visualize('data/geoaq/MSMARCO/analyzed_question_new_parser.json', 'MSMARCO')
# write_ngrams_as_csv(ms_marco_unique, 5, 'ngrams/MS_MARCO_UNIQUE.csv')
# logging.info('analyzing MS MARCO (ALL) dataset...')
# ms_marco_all = read_msmarco_code('data/geoaq/all_codes_msmarco.csv')
# write_ngrams_as_csv(ms_marco_all, 5, 'ngrams/MS_MARCO_ALL.csv')
# d4 = generate_vectors_ms_marco(ms_marco_all)

# comparing different dataset by generating the scatterplots...
logging.info('Comparing datasets (random subset: 200, dimension: 16d)...')
dict_datasets = {}
dict_datasets['GeoAnalytical'] = {'dataset': get_random_sample(d2['vectors'], 200), 'color': 'blue'}
dict_datasets['GeoQA GS'] = {'dataset': get_random_sample(d1['vectors'], 200), 'color': 'green'}
# dict_datasets['MS MARCO (unique)'] = {'dataset': get_random_sample(d3['vectors'], 200), 'color': 'orange'}
# dict_datasets['MS MARCO (all)'] = {'dataset': get_random_sample(d4['vectors'], 200), 'color': 'red'}
dict_datasets['MS MARCO'] = {'dataset': get_random_sample(d3['vectors'], 200), 'color': 'orange'}
write_2d_scatter_comparison_plot(dict_datasets, 'graphs/2d_encoding_comparison.png', 16)

logging.info('Comparing datasets (all, dimension: 16d)...')
dict_datasets2 = {}
dict_datasets2['GeoAnalytical'] = {'dataset': d2['vectors'], 'color': 'blue'}
dict_datasets2['GeoQA GS'] = {'dataset': d1['vectors'], 'color': 'green'}
dict_datasets2['MS MARCO'] = {'dataset': d3['vectors'], 'color': 'orange'}
write_2d_scatter_comparison_plot(dict_datasets2, 'graphs/2d_encoding_comparison-without-sampling.png', 16)

logging.info('Comparing datasets (random subset: 200, dimension: 9d)...')
dict_datasets = {}
dict_datasets['GeoAnalytical'] = {'dataset': translate_vectors_to_9d(get_random_sample(d2['vectors'], 200)),
                                  'color': 'blue'}
dict_datasets['GeoQA GS'] = {'dataset': translate_vectors_to_9d(get_random_sample(d1['vectors'], 200)),
                             'color': 'green'}
# dict_datasets['MS MARCO (unique)'] = {'dataset': translate_vectors_to_9d(get_random_sample(d3['vectors'], 200)), 'color': 'orange'}
dict_datasets['MS MARCO'] = {'dataset': translate_vectors_to_9d(get_random_sample(d3['vectors'], 200)),
                             'color': 'orange'}
# dict_datasets['MS MARCO (all)'] = {'dataset': translate_vectors_to_9d(get_random_sample(d4['vectors'], 200)), 'color': 'red'}
write_2d_scatter_comparison_plot(dict_datasets, 'graphs/2d_encoding_comparison_9d.png', 9)

logging.info('Comparing datasets (all, dimension: 9d)...')
dict_datasets2 = {}
dict_datasets2['GeoAnalytical'] = {'dataset': translate_vectors_to_9d(d2['vectors']), 'color': 'blue'}
dict_datasets2['GeoQA GS'] = {'dataset': translate_vectors_to_9d(d1['vectors']), 'color': 'green'}
dict_datasets2['MS MARCO'] = {'dataset': translate_vectors_to_9d(d3['vectors']), 'color': 'orange'}
write_2d_scatter_comparison_plot(dict_datasets2, 'graphs/2d_encoding_comparison-without-sampling_9d.png', 9)

logging.info('source-based and category based comparison of GeoAnalytical questions')
generate_based_on_category(
    'data/geoaq/Simon-dataset/with source and q_type/analyzed_question_new_parser_source_and_type.json')

logging.info('clustering based on encodings to find types of questions in all datasets...')
dict_clustering = merge_dicts(d1, d2)
dict_clustering = merge_dicts(dict_clustering, d3)
encoding_cluster(dict_clustering)
emb_clustering('data/geoaq/embeddings')

# todo combined_clusters -- get the result of encoding cluster
# filter ms marco, and geoqa questions in a cluster that is mostly geoanalytical
# run embedding cluster over thoese
# find the most similar ones to geoanalytical questions...
