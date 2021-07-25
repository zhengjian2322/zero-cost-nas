def config_to_genotype(config, blocks_in_cell=4):
    from collections import namedtuple
    Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
    # Normal Cell
    normal = list()
    for i in range(blocks_in_cell):
        normal.append((config['normal_op%d' % (2 * i)], config['normal_connection%d' % (2 * i)]))
        normal.append((config['normal_op%d' % (2 * i + 1)], config['normal_connection%d' % (2 * i + 1)]))
    normal_concat = [2, 3, 4, 5]
    # Reduction Cell
    reduce = list()
    for i in range(blocks_in_cell):
        reduce.append((config['reduce_op%d' % (2 * i)], config['reduce_connection%d' % (2 * i)]))
        reduce.append((config['reduce_op%d' % (2 * i + 1)], config['reduce_connection%d' % (2 * i + 1)]))
    reduce_concat = [2, 3, 4, 5]
    return Genotype(normal=normal, normal_concat=normal_concat, reduce=reduce, reduce_concat=reduce_concat)
