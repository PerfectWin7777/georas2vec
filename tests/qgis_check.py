# Vrai test QGIS (PyQGIS 3.34, sans interface) d'un GeoPackage produit par pipeline.py
#   QT_QPA_PLATFORM=offscreen python3.12 qgis_test.py <carte.gpkg> <rendu.png>
import sys
from qgis.core import (QgsApplication, QgsVectorLayer, QgsProject, QgsMapSettings, QgsMapRendererParallelJob,
                       QgsGeometry, QgsWkbTypes)
from qgis.PyQt.QtCore import QSize
from qgis.PyQt.QtGui import QColor

gpkg, png = sys.argv[1], sys.argv[2]
app = QgsApplication([], False); app.initQgis()
ok = True
def check(cond, msg):
    global ok
    print(("OK   " if cond else "ÉCHEC"), msg); ok &= bool(cond)

import sqlite3
_tabs = {r[0] for r in sqlite3.connect(gpkg).execute("SELECT table_name FROM gpkg_contents")}
EN = "geology" in _tabs
U, FL, LG = ("geology", "geology_faults", "geology_legend") if EN else ("geologie", "geologie_failles", "geologie_legende")
C = (dict(nom="name", lithologie="lithology", source_nom="source", confiance="confidence", lu="read on the map") if EN
     else dict(nom="nom", lithologie="lithologie", source_nom="source_nom", confiance="confiance", lu="code lu sur la carte"))
layers = []
for name, geom in ((U, QgsWkbTypes.PolygonGeometry), (FL, QgsWkbTypes.LineGeometry)):
    lyr = QgsVectorLayer(f"{gpkg}|layername={name}", name, "ogr")
    check(lyr.isValid(), f"couche {name} chargée")
    if not lyr.isValid():
        continue
    msg, styled = lyr.loadDefaultStyle()            # style QML stocké dans la table layer_styles du GeoPackage
    check(styled, f"{name} : style intégré chargé automatiquement ({msg})")
    check(lyr.geometryType() == geom, f"{name} : type de géométrie {QgsWkbTypes.geometryDisplayString(lyr.geometryType())}")
    check(lyr.crs().isValid(), f"{name} : SCR {lyr.crs().authid() or lyr.crs().description()}")
    n = lyr.featureCount(); check(n > 0 or name == FL, f"{name} : {n} entités")
    bad = sum(1 for f in lyr.getFeatures() if not f.geometry().isGeosValid())
    check(bad == 0, f"{name} : géométries invalides = {bad}")
    print("      champs :", [f.name() for f in lyr.fields()])
    r = lyr.renderer()
    print("      rendu :", r.type(), (f"{len(r.categories())} catégories" if hasattr(r, "categories") else ""))
    if name == U:
        need = {"code", C["nom"], "age", C["lithologie"], C["source_nom"], C["confiance"], "suggestions"}
        check(need <= {f.name() for f in lyr.fields()}, "colonnes de nommage présentes")
        # topologie : pas de chevauchement entre polygones voisins (somme des surfaces = surface de l'union)
        gs = [f.geometry() for f in lyr.getFeatures()]
        u = QgsGeometry.unaryUnion(gs)
        s_sum = sum(g.area() for g in gs); s_u = u.area()
        check(abs(s_sum - s_u) / s_u < 1e-6, f"{name} : aucun chevauchement entre polygones (écart {abs(s_sum - s_u) / s_u:.2e})")
        # filtre attributaire comme le ferait un utilisateur QGIS
        lyr.setSubsetString(f"\"{C['source_nom']}\" = '{C['lu']}'")
        print(f"      filtre {C['source_nom']}='{C['lu']}' :", lyr.featureCount(), "entités"); lyr.setSubsetString("")
        ex = next(lyr.getFeatures())
        print("      exemple :", {k: ex[k] for k in ("code", C["nom"], "age", C["source_nom"], C["confiance"])})
    layers.append(lyr)

leg = QgsVectorLayer(f"{gpkg}|layername={LG}", LG, "ogr")
check(leg.isValid() and leg.featureCount() > 0, f"table {LG} : {leg.featureCount() if leg.isValid() else 0} lignes")

if layers:
    QgsProject.instance().addMapLayers(layers)
    ms = QgsMapSettings(); ms.setLayers(layers[::-1]); ms.setDestinationCrs(layers[0].crs())
    ext = layers[0].extent(); ms.setExtent(ext); ms.setOutputSize(QSize(1400, int(1400 * ext.height() / ext.width())))
    ms.setBackgroundColor(QColor("white"))
    job = QgsMapRendererParallelJob(ms); job.start(); job.waitForFinished()
    img = job.renderedImage(); check(img.save(png), f"rendu carte -> {png}")
    # le rendu doit montrer les couleurs de la carte (pas un aplat gris : style mal relié aux champs)
    from PIL import Image
    import numpy as np
    a = np.asarray(Image.open(png).convert("RGB")).reshape(-1, 3)
    a = a[(a.max(1) - a.min(1)) > 25]                     # pixels colorés (ni blanc, ni gris, ni noir)
    check(len(a) > 0.2 * img.width() * img.height(), f"rendu en couleurs ({len(a) / (img.width() * img.height()):.0%} de pixels colorés)")
    QgsProject.instance().write(png.rsplit(".", 1)[0] + ".qgz")
    print("      projet QGIS écrit :", png.rsplit(".", 1)[0] + ".qgz")
print("\nRÉSULTAT QGIS :", "TOUT OK" if ok else "PROBLÈMES")
app.exitQgis()
