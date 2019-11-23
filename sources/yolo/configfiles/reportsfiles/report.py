import json
import csv


def load_data(jsonfile):
    """ load data from a json file """

    with open(jsonfile, 'r') as fp:
        data = json.load(fp)

    return data


def fill_data(data, iou, fp, no_detect, type):
    if len(data['boxes']) > 0:
        for keybox in data['boxes'].keys():
            iou_value = data['boxes'][keybox]["IoU"]

            if iou_value > 0.2:
                iou[type].append(iou_value)
            else:
                fp[type] += 1
    else:
        no_detect[type] += 1

    return iou, fp, no_detect


def nightly_main():
    """ create csv with nightly results """

    data = load_data('result.json')
    labels = load_data('nightly_classification.json')

    night, day = labels['night'], labels['day']

    fp = {'night':0, 'day':0}
    no_detect = {'night':0, 'day':0}
    iou = {'night':[], 'day':[]}

    for key in data.keys():
        if key in night:
            iou, fp, no_detect = fill_data(data[key], iou, fp, no_detect, 'night')
        else:
            iou, fp, no_detect = fill_data(data[key], iou, fp, no_detect, 'day')

    night_iou_mean = sum(iou['night']) / len(iou['night'])
    day_iou_mean = sum(iou['day']) / len(iou['day'])

    with open('night_results.csv', mode='w') as nr:
        writer = csv.writer(nr, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        writer.writerow(['type', 'Iou_Min', 'IoU_Max', 'IoU_Mean', 'total_images','no_detected', 'false_positive'])
        writer.writerow(['night', str(min(iou['night'])), str(max(iou['night'])), str(night_iou_mean), len(night), str(no_detect["night"]), str(fp['night'])])
        writer.writerow(['day', str(min(iou['day'])), str(max(iou['day'])), str(day_iou_mean), len(day), str(no_detect["day"]), str(fp['day'])])


def occluded_main():
    """ create csv with occluded results """

    data = load_data('result.json')
    labels = load_data('occluded_classification.json')

    occluded, notoccluded = labels['occluded'], labels['notoccluded']

    fp = {'occluded':0, 'notoccluded':0}
    no_detect = {'occluded':0, 'notoccluded':0}
    iou = {'occluded':[], 'notoccluded':[]}

    for key in data.keys():
        if key in occluded:
            iou, fp, no_detect = fill_data(data[key], iou, fp, no_detect, 'occluded')
        else:
            iou, fp, no_detect = fill_data(data[key], iou, fp, no_detect, 'notoccluded')

    occluded_iou_mean = sum(iou['occluded']) / len(iou['occluded'])
    notoccluded_iou_mean = sum(iou['notoccluded']) / len(iou['notoccluded'])

    with open('occluded_results.csv', mode='w') as nr:
        writer = csv.writer(nr, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        writer.writerow(['type', 'Iou_Min', 'IoU_Max', 'IoU_Mean', 'total_images', 'no_detected', 'false_positive'])
        writer.writerow(['occluded', str(min(iou['occluded'])), str(max(iou['occluded'])), str(occluded_iou_mean), len(occluded), str(no_detect["occluded"]), str(fp['occluded'])])
        writer.writerow(['notoccluded', str(min(iou['notoccluded'])), str(max(iou['notoccluded'])), str(notoccluded_iou_mean), len(notoccluded), str(no_detect["notoccluded"]), str(fp['notoccluded'])])


def front_main():
    """ create csv with front results """

    data = load_data('result.json')
    labels = load_data('front_classification.json')

    front, nofront = labels['front'], labels['nofront']

    fp = {'front':0, 'nofront':0}
    no_detect = {'front':0, 'nofront':0}
    iou = {'front':[], 'nofront':[]}

    for key in data.keys():
        if key in front:
            iou, fp, no_detect = fill_data(data[key], iou, fp, no_detect, 'front')
        else:
            iou, fp, no_detect = fill_data(data[key], iou, fp, no_detect, 'nofront')

    front_iou_mean = sum(iou['front']) / len(iou['front'])
    nofront_iou_mean = sum(iou['nofront']) / len(iou['nofront'])

    with open('front_results.csv', mode='w') as nr:
        writer = csv.writer(nr, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        writer.writerow(['type', 'Iou_Min', 'IoU_Max', 'IoU_Mean', 'total_images', 'no_detected', 'false_positive'])
        writer.writerow(['front', str(min(iou['front'])), str(max(iou['front'])), str(front_iou_mean), len(front), str(no_detect["front"]), str(fp['front'])])
        writer.writerow(['nofront', str(min(iou['nofront'])), str(max(iou['nofront'])), str(nofront_iou_mean), len(nofront), str(no_detect["nofront"]), str(fp['nofront'])])


if __name__ == "__main__":
    nightly_main()
    occluded_main()
    front_main()
