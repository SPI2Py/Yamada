node_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94',
               '95', '96', '97']

node_positions = {'0': (5, 1, 6),
                  '1': (6, 7, 6),
                  '2': (2, 1, 6),
                  '3': (2, 2, 5),
                  '4': (3, 7, 6),
                  '5': (3, 2, 1),
                  '6': (2, 6, 5),
                  '7': (3, 3, 2),
                  '8': (7, 7, 2),
                  '9': (6, 4, 6),
                  '10': (3, 5, 2),
                  '11': (4, 6, 0),
                  '12': (1, 3, 2),
                  '13': (2, 7, 6),
                  '14': (5, 7, 2),
                  '15': (3, 6, 3),
                  '16': (6, 2, 2),
                  '17': (7, 2, 2),
                  '18': (5, 8, 6),
                  '19': (2, 6, 0),
                  '20': (4, 6, 2),
                  '21': (4, 1, 6),
                  '22': (6, 2, 4),
                  '23': (7, 3, 2),
                  '24': (6, 7, 3),
                  '25': (2, 2, 2),
                  '26': (2, 6, 2),
                  '27': (5, 2, 1),
                  # '28': (2, 5, 4),
                  '29': (6, 3, 8),
                  '30': (2, 4, 2),
                  '31': (6, 2, 6),
                  '32': (2, 3, 6),
                  '33': (7, 3, 4),
                  '34': (7, 2, 6),
                  '35': (6, 7, 5),
                  '36': (2, 2, 4),
                  # '37': (2, 6, 4),
                  '38': (1, 2, 1),
                  '39': (6, 6, 6),
                  '40': (6, 4, 8),
                  '41': (5, 6, 0),
                  '42': (4, 8, 6),
                  '43': (2, 6, 6),
                  '44': (1, 3, 1),
                  '45': (3, 1, 6),
                  '46': (3, 3, 3),
                  '47': (3, 4, 2),
                  '48': (3, 6, 5),
                  '49': (6, 5, 2),
                  '50': (6, 6, 1),
                  '51': (5, 7, 6),
                  '52': (2, 3, 3),
                  '53': (6, 7, 2),
                  '54': (6, 3, 7),
                  '55': (4, 6, 3),
                  '56': (6, 6, 3),
                  '57': (5, 5, 2),
                  '58': (3, 6, 0),
                  # '59': (2, 3, 5),
                  # '60': (2, 4, 4),
                  '61': (6, 7, 4),
                  '62': (7, 8, 2),
                  '63': (2, 2, 6),
                  '64': (6, 5, 6),
                  '65': (3, 2, 2),
                  '66': (3, 6, 2),
                  '67': (6, 2, 1),
                  '68': (6, 4, 7),
                  '69': (5, 6, 2),
                  '70': (6, 3, 2),
                  '71': (4, 2, 1),
                  '72': (1, 2, 2),
                  '73': (4, 7, 6),
                  '74': (3, 6, 4),
                  '75': (2, 2, 1),
                  '76': (6, 3, 4),
                  '77': (6, 6, 0),
                  '78': (2, 6, 1),
                  '79': (3, 4, 3),
                  '80': (2, 3, 2),
                  '81': (3, 6, 6),
                  '82': (6, 2, 5),
                  '83': (6, 4, 2),
                  '84': (7, 1, 6),
                  '85': (7, 3, 3),
                  '86': (6, 8, 2),
                  '87': (7, 6, 2),
                  '88': (5, 6, 6),
                  '89': (2, 2, 3),
                  '90': (2, 5, 2),
                  '91': (6, 1, 6),
                  '92': (6, 3, 6),
                  '93': (6, 6, 2),
                  # '94': (2, 3, 4),
                  '95': (2, 4, 3),
                  # UPDATED
                  # '59': (2, 3, 5),
                  # '60': (2, 4, 4),
                  # '94': (2, 3, 4),
                  # '28': (2, 5, 4),
                  # '37': (2, 6, 4),
                  '59': (2, 3, 5),
                  '94': (2, 3, 4),
                  '96': (2.25, 3.25, 2.5),
                  '97': (1.75, 3.75, 2.5),
                  '60': (2, 4, 4),
                  '28': (2, 5, 4),
                  '37': (2, 6, 4),

                  # 'a2': (),
                  # 'a3': ()
                  }

edges = [('25', '80'),
         ('80', '30'),
         ('30', '95'),
         ('95', '52'),
         ('52', '89'),
         ('89', '36'),
         ('36', '3'),
         ('3', '63'),
         ('25', '65'),
         ('65', '7'),
         ('7', '46'),
         ('46', '79'),
         ('79', '47'),
         ('47', '10'),
         ('10', '90'),
         ('90', '26'),
         ('25', '72'),
         ('72', '12'),
         ('12', '44'),
         ('44', '38'),
         ('38', '75'),
         ('75', '5'),
         ('5', '71'),
         ('71', '27'),
         ('27', '67'),
         ('67', '16'),
         ('63', '32'),
         ('32', '59'),
         # ('59', '94'),
         # ('94', '60'),
         # ('60', '28'),
         # ('28', '37'),
         ('37', '6'),
         ('6', '43'),
         ('63', '2'),
         ('2', '45'),
         ('45', '21'),
         ('21', '0'),
         ('0', '91'),
         ('91', '84'),
         ('84', '34'),
         ('34', '31'),
         ('26', '66'),
         ('66', '20'),
         ('20', '55'),
         ('55', '15'),
         ('15', '74'),
         ('74', '48'),
         ('48', '81'),
         ('81', '43'),
         ('26', '78'),
         ('78', '19'),
         ('19', '58'),
         ('58', '11'),
         ('11', '41'),
         ('41', '77'),
         ('77', '50'),
         ('50', '93'),
         ('43', '13'),
         ('13', '4'),
         ('4', '73'),
         ('73', '42'),
         ('42', '18'),
         ('18', '51'),
         ('51', '88'),
         ('88', '39'),
         ('16', '17'),
         ('17', '23'),
         ('23', '85'),
         ('85', '33'),
         ('33', '76'),
         ('76', '22'),
         ('22', '82'),
         ('82', '31'),
         ('16', '70'),
         ('70', '83'),
         ('83', '49'),
         ('49', '57'),
         ('57', '69'),
         ('69', '14'),
         ('14', '53'),
         ('53', '86'),
         ('86', '62'),
         ('62', '8'),
         ('8', '87'),
         ('87', '93'),
         ('31', '92'),
         ('92', '54'),
         ('54', '29'),
         ('29', '40'),
         ('40', '68'),
         ('68', '9'),
         ('9', '64'),
         ('64', '39'),
         ('93', '56'),
         ('56', '24'),
         ('24', '61'),
         ('61', '35'),
         ('35', '1'),
         ('1', '39'),
         # UPDATED
         # ('59', '94'),
         # ('94', '60'),
         # ('60', '28'),
         # ('28', '37')
         ('59', '94'),
         ('94', '96'),
         ('96', '97'),
         ('97', '60'),
         ('60', '28'),
         ('28', '37')]
