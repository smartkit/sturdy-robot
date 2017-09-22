import gzip, re, os
from ynlib.files import ReadFromFile, WriteToFile
from ynlib.system import Execute
from xml.dom.minidom import parse, parseString


def SVGToPNGInMemory(svgPath, newWidth, backgroundColor,ploygonFillColor):

    tempPath = os.path.join(self.rootFolder, 'data')
    fileNameRoot = 'temp_' + str(image.getID())

    if svgPath.lower().endswith('svgz'):
        svg = gzip.open(svgPath, 'rb').read()
    else:
        svg = ReadFromFile(svgPath)

    xmldoc = parseString(svg)

    width = float(xmldoc.getElementsByTagName("svg")[0].attributes['width'].value.split('px')[0])
    height = float(xmldoc.getElementsByTagName("svg")[0].attributes['height'].value.split('px')[0])

    newHeight = int(newWidth / width * height) 

    xmldoc.getElementsByTagName("svg")[0].attributes['width'].value = '%spx' % newWidth
    xmldoc.getElementsByTagName("svg")[0].attributes['height'].value = '%spx' % newHeight


    xmldoc.getElementsByTagName("ploygon")[0].attributes['fill'].value = ploygonFillColor

    WriteToFile(os.path.join(tempPath, fileNameRoot + '.svg'), xmldoc.toxml())
    Execute('convert -background "%s" %s %s' % (backgroundColor, os.path.join(tempPath, fileNameRoot + '.svg'), os.path.join(tempPath, fileNameRoot + '.png')))

    png = open(os.path.join(tempPath, fileNameRoot + '.png'), 'rb').read()

    os.remove(os.path.join(tempPath, fileNameRoot + '.png'))
    os.remove(os.path.join(tempPath, fileNameRoot + '.svg'))

    return png

if __name__ == '__main__':
    main()