from utils.logger import LogVisualizer
vis = LogVisualizer()
vis.sessions('logs/custom_resnet50.log')
vis.add('logs/custom_resnet50.log', session=0)
vis.plot('train', 'x.data.iter', 'x.data.loss.T', smoothness=1000)
# vis.plot('val', 'x.data.iter', 'x.data.box["all"]', smoothness=100)
# vis.plot('val', 'x.data.epoch', 'x.data.box["all"]',smoothness=10)
# vis.sessions('logs/yolact_resnet50.log')
# vis.add('logs/yolact_base.log', session=0)
# vis.plot('val', 'x.data.epoch', 'x.data.box.all')

