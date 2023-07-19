from ..Qt import QtGui, QtCore
from .Exporter import Exporter
from ..parametertree import Parameter
from .. import PlotItem

__all__ = ['CSVExporter']
    
    
class CSVExporter(Exporter):
    Name = "CSV from plot data"
    windows = []
    def __init__(self, item):
        Exporter.__init__(self, item)
        self.params = Parameter(name='params', type='group', children=[
            {'name': 'separator', 'type': 'list', 'value': 'comma', 'values': ['comma', 'tab']},
            {'name': 'precision', 'type': 'int', 'value': 10, 'limits': [0, None]},
            {'name': 'columnMode', 'type': 'list', 'values': ['(x,y) per plot', '(x,y,y,y) for all plots']}
        ])
        
    def parameters(self):
        return self.params
    
    def export(self, fileName=None):
        
        if not isinstance(self.item, PlotItem):
            raise Exception("Must have a PlotItem selected for CSV export.")

        if fileName is None:
            self.fileSaveDialog(filter=["*.csv", "*.tsv"])
            return

        with open(fileName, 'w') as fd:
            data = []
            header = []

            appendAllX = self.params['columnMode'] == '(x,y) per plot'

            for i, c in enumerate(self.item.curves):
                cd = c.getData()
                if cd[0] is None:
                    continue
                data.append(cd)
                if hasattr(c, 'implements') and c.implements('plotData') and c.name() is not None:
                    name = c.name().replace('"', '""') + '_'
                    xName, yName = f'"{name}x"', f'"{name}y"'
                else:
                    xName = 'x%04d' % i
                    yName = 'y%04d' % i
                if appendAllX or i == 0:
                    header.extend([xName, yName])
                else:
                    header.extend([yName])

            sep = ',' if self.params['separator'] == 'comma' else '\t'
            fd.write(sep.join(header) + '\n')
            i = 0
            numFormat = '%%0.%dg' % self.params['precision']
            numRows = max(len(d[0]) for d in data)
            for i in range(numRows):
                for j, d in enumerate(data):
                    if appendAllX or j == 0:
                        if d is not None and i < len(d[0]):
                            fd.write(numFormat % d[0][i] + sep)
                        else:
                            fd.write(f' {sep}')

                                # write y value 
                    if d is not None and i < len(d[1]):
                        fd.write(numFormat % d[1][i] + sep)
                    else:
                        fd.write(f' {sep}')
                fd.write('\n')

CSVExporter.register()        
                
        
