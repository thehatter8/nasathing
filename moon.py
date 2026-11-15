"""
Herramienta de línea de comandos para datos MOLA (Marte) del PDS / ODE.

Subcomandos:
  buscar        consulta ODE, guarda las URLs y descarga los archivos encontrados
  descargar     descarga archivos por URL (con reintentos y reanudación)
  inspeccionar  metadatos y estadísticas de un raster (.lbl PDS3 o .tif)
  procesar      inspecciona y genera GeoTIFF para un DEM o una carpeta de etiquetas .lbl
  selftest      prueba con datos sintéticos

Dependencias: pip install rasterio scipy requests numpy
"""
import argparse
import hashlib
import importlib
import json
import math
import re
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import unquote, urlsplit

try:
    import numpy as np
except ImportError:
    sys.exit("Falta numpy. Instala con: pip install rasterio scipy requests numpy")

# Radio IAU usado solo en los datos sintéticos. Los datos reales usan su CRS.
R_MARTE_M = 3_396_190.0

# ODE REST Interface Manual v2.1.6 (2024).
ODE_ENDPOINT = "https://oderest.rsl.wustl.edu/live2/"
EXT_UTILES = (".img", ".lbl", ".tif", ".tiff", ".xml", ".jp2", ".hdr")
DESTINO_CRUDO = "datos_marte/crudo"


# --------------------------------------------------------------------------
# utilidades
# --------------------------------------------------------------------------
def _importar(nombre, paquete=None):
    """Importa un módulo con un mensaje de error amable si falta."""
    try:
        return importlib.import_module(nombre)
    except ImportError:
        sys.exit(f"Falta {paquete or nombre}. Instala con: "
                 "pip install rasterio scipy requests numpy")


def _importar_rasterio():
    rasterio = _importar("rasterio")
    from rasterio.enums import Resampling
    return rasterio, Resampling


def _recolectar_urls(productos, acc):
    """Reúne los archivos que ODE clasifica como Product."""
    for producto in productos:
        if not isinstance(producto, dict):
            raise ValueError("ODE devolvió Product en un formato desconocido.")
        contenedor = producto.get("Product_files") or {}
        if not isinstance(contenedor, dict):
            raise ValueError("ODE devolvió Product_files en un formato desconocido.")
        archivos = contenedor.get("Product_file") or []
        if isinstance(archivos, dict):
            archivos = [archivos]
        if not isinstance(archivos, list):
            raise ValueError("ODE devolvió Product_file en un formato desconocido.")
        for archivo in archivos:
            if not isinstance(archivo, dict):
                raise ValueError("ODE devolvió un archivo de producto inválido.")
            if str(archivo.get("Type", "")).strip().casefold() != "product":
                continue
            u = archivo.get("URL")
            if isinstance(u, str):
                u = u.strip()
                partes = urlsplit(u)
                if (partes.scheme in {"http", "https"} and partes.netloc
                        and partes.path.lower().endswith(EXT_UTILES)):
                    acc.add(u)


def _factor_lineal(crs):
    """Metros por unidad lineal; nunca se inventan unidades desconocidas."""
    try:
        k = float(crs.linear_units_factor[1])
    except (AttributeError, ValueError) as e:
        raise ValueError("No se pueden determinar las unidades lineales del CRS.") from e
    if not math.isfinite(k) or k <= 0:
        raise ValueError("Unidades lineales inválidas.")
    return k


def _elipsoide(crs):
    if crs is None or not (crs.is_geographic or crs.is_projected):
        raise ValueError("Se requiere un CRS geográfico o proyectado explícito.")
    m = re.search(r'(?:SPHEROID|ELLIPSOID)\["[^"]+",\s*([\d.eE+-]+),\s*([\d.eE+-]+)',
                  crs.to_wkt())
    if not m:
        raise ValueError("El CRS no declara un elipsoide utilizable.")
    a, inv_f = map(float, m.groups())
    f = 0.0 if inv_f == 0 else 1.0 / inv_f
    if not math.isfinite(a) or a <= 0 or not 0 <= f < 1:
        raise ValueError("Elipsoide inválido.")
    return a, f * (2.0 - f)


def _crs_geografico(crs):
    """CRS base del mismo planeta, compatible también con Rasterio 1.4."""
    if crs.is_geographic:
        return crs
    datos = crs.to_dict(projjson=True)
    if datos.get("type") == "BoundCRS":
        datos = datos["source_crs"]
    base = datos.get("base_crs")
    if not isinstance(base, dict):
        raise ValueError("El CRS no declara un sistema geográfico base.")
    return type(crs).from_dict(dict(base, type="GeographicCRS"))


def _validar_grilla(transform, crs, cuerpo=None):
    a, _ = _elipsoide(crs)
    if not all(math.isfinite(v) for v in transform[:6]):
        raise ValueError("Transformación no finita.")
    if transform.b != 0 or transform.d != 0 or transform.a <= 0 or transform.e >= 0:
        raise ValueError("Se requiere una grilla norte-arriba sin rotación. Reproyecta primero.")
    limites = {"marte": (3.3e6, 3.5e6), "tierra": (6.3e6, 6.5e6)}
    if cuerpo is not None:
        if cuerpo not in limites or not limites[cuerpo][0] <= a <= limites[cuerpo][1]:
            raise ValueError(f"El CRS no corresponde al cuerpo solicitado ({cuerpo}). "
                             "Usa un CRS de Marte para MOLA; EPSG:4326 es de la Tierra.")
    # GDAL's geographic coordinates for this workflow must be longitude/latitude in degrees.
    factor_angular = _crs_geografico(crs).units_factor[1]
    if not math.isclose(factor_angular, math.pi / 180, rel_tol=1e-10):
        raise ValueError("El CRS geográfico debe usar grados.")


def _factor_vertical(unidad):
    u = (unidad or "m").strip().lower()
    factores = {"m": 1., "meter": 1., "meters": 1., "metre": 1., "metres": 1.,
                "km": 1000., "kilometer": 1000., "kilometers": 1000.,
                "kilometre": 1000., "kilometres": 1000.,
                "ft": .3048, "foot": .3048, "feet": .3048,
                "us_survey_foot": 1200. / 3937.}
    if u not in factores:
        raise ValueError(f"Unidad vertical desconocida: {unidad!r}; indica --unidad-z.")
    return factores[u]


def _entero_positivo(v):
    n = int(v)
    if n < 1:
        raise argparse.ArgumentTypeError("Debe ser un entero mayor que cero.")
    return n


def _real_positivo(v):
    n = float(v)
    if not math.isfinite(n) or n <= 0:
        raise argparse.ArgumentTypeError("Debe ser un número finito mayor que cero.")
    return n


def _escribir_texto(ruta, texto):
    """Reemplazo atómico: un fallo no destruye un resultado anterior."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=ruta.parent,
                                     suffix=".tmp", delete=False) as f:
        tmp = Path(f.name)
        try:
            f.write(texto)
        except BaseException:
            f.close()
            tmp.unlink(missing_ok=True)
            raise
    try:
        tmp.replace(ruta)
    finally:
        tmp.unlink(missing_ok=True)


# --------------------------------------------------------------------------
# 1) buscar
# --------------------------------------------------------------------------
def cmd_buscar(args):
    requests = _importar("requests")
    params = {
        "query": "product",
        "target": args.target,
        "ihid": args.ihid,
        "iid": args.iid,
        "pt": args.pt,
        "results": args.results,
        "output": "JSON",
        "limit": args.limit,
        "offset": args.offset,
    }
    if args.offset < 0:
        raise ValueError("--offset no puede ser negativo (ODE empieza en 0).")
    if not args.results or set(args.results.lower()) - set("copmfl") or "f" not in args.results.lower():
        raise ValueError("--results debe incluir f (archivos); opciones: c, o, p, m, f, l.")
    if args.pdsid:
        params["pdsid"] = args.pdsid
    elif args.pt.upper() == "MEGDR" and not args.todos_tipos:
        params["pdsid"] = "MEGT*"  # topografía, no radio/areoide/recuentos
    for kv in args.extra or []:
        k, sep, v = kv.partition("=")
        k = k.strip().lower()
        if not sep or not k or not v:
            raise ValueError(f"Parámetro extra inválido: {kv!r}; usa clave=valor.")
        if k in params or k in {"q", "r", "o", "t", "jsonp"}:
            raise ValueError(f"--extra no puede sobrescribir {k}; usa su opción explícita.")
        params[k] = v

    salida = Path(args.salida)
    salida.parent.mkdir(parents=True, exist_ok=True)
    urls = set()
    vistos = set()
    for pagina in range(args.paginas):
        with requests.get(args.endpoint, params=params, timeout=(15, 120)) as r:
            print("URL consultada:", r.url)
            raw = salida.with_suffix(".raw.txt" if pagina == 0 else f".pagina{pagina + 1}.raw.txt")
            _escribir_texto(raw, r.text)
            r.raise_for_status()
            try:
                datos = r.json()
            except ValueError as e:
                raise ValueError(f"ODE no devolvió JSON; revisa {raw}.") from e
        if not isinstance(datos, dict):
            raise ValueError(f"Respuesta ODE inválida; revisa {raw}.")
        root = datos.get("ODEResults", datos)
        if not isinstance(root, dict):
            raise ValueError(f"Respuesta ODE inválida; revisa {raw}.")
        campos = {k.lower(): v for k, v in root.items()}
        if str(campos.get("status", "")).upper() != "SUCCESS":
            raise ValueError(f"ODE: {campos.get('error', 'respuesta sin Status=Success')}")
        contenedor = campos.get("products") or {}
        if not isinstance(contenedor, dict):
            raise ValueError("ODE devolvió Products en un formato desconocido.")
        productos = contenedor.get("Product", [])
        if isinstance(productos, dict):
            productos = [productos]
        if not isinstance(productos, list):
            raise ValueError("ODE devolvió Products en un formato desconocido.")
        huella = hashlib.sha256(json.dumps(productos, sort_keys=True).encode()).hexdigest()
        if productos and huella in vistos:
            raise ValueError("ODE repitió una página; se detiene para evitar resultados incompletos.")
        vistos.add(huella)
        _recolectar_urls(productos, urls)
        print(f"Página {pagina + 1}: {len(productos)} productos")
        if len(productos) < args.limit:
            break
        params["offset"] += len(productos)
    else:
        print("Se alcanzó --paginas; puede haber más productos. Próximo --offset:", params["offset"])
    if args.filtro:
        urls = {u for u in urls if args.filtro.lower() in u.lower()}
    urls = sorted(urls)
    _escribir_texto(salida, "\n".join(urls) + ("\n" if urls else ""))

    print(len(urls), "URLs de archivos de producto guardadas en", salida)
    for u in urls[:30]:
        print("  ", u)
    if not urls:
        print("Sin URLs. Revisa", salida.with_suffix(".raw.txt"),
              "y los parámetros (manual ODE REST). Puedes añadir parámetros con --extra clave=valor")
        return
    if not args.solo_lista:
        _descargar_urls(requests, urls, args.destino, args.reintentos, args.forzar)


# --------------------------------------------------------------------------
# 2) descargar
# --------------------------------------------------------------------------
class _ColisionDescarga(ValueError):
    """El nombre local ya pertenece a otro archivo."""


def _nombre_de_url(u):
    partes = urlsplit(u)
    if partes.scheme.lower() not in {"http", "https"} or not partes.netloc:
        raise ValueError(f"URL HTTP(S) inválida: {u}")
    base = unquote(partes.path.rsplit("/", 1)[-1])
    if (not base or base in {".", ".."} or re.search(r'[<>:"/\\|?*\x00-\x1f]', base)
            or base.endswith((" ", ".")) or len(base) > 200
            or re.match(r'(?i)^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)', base)):
        raise ValueError(f"Nombre de archivo no seguro o incompatible con Windows: {base!r}")
    return base


def _ruta_origen(archivo):
    return archivo.parent / ".descargas" / (archivo.name + ".txt")


def _migrar_registros_url(destino):
    """Limpia los registros del descargador anterior y conserva el origen de los datos."""
    raiz = destino.resolve()
    for carpeta in [destino, *destino.glob("fuente_*")]:
        if not carpeta.is_dir() or not carpeta.resolve().is_relative_to(raiz):
            continue
        for marca in carpeta.glob("*.url"):
            archivo = marca.with_suffix("")
            try:
                if marca.is_symlink() or not marca.is_file() or marca.stat().st_size > 16384:
                    continue
                url = marca.read_text(encoding="utf-8").strip()
                # La versión anterior guardaba una URL sola, con el mismo nombre del archivo.
                if "\n" in url or "\r" in url or _nombre_de_url(url).casefold() != archivo.name.casefold():
                    continue
                registro = _ruta_origen(archivo)
                if archivo.is_file() and not registro.exists():
                    _escribir_texto(registro, url)
                marca.unlink()
            except ValueError:
                continue  # accesos directos reales u otros archivos del usuario
            except OSError as e:
                print(f"No se pudo migrar el registro {marca}: {e}")


def _nombre_local(u, destino, usados):
    """Conserva los nombres PDS y evita sobrescribir archivos de otro origen."""
    base = _nombre_de_url(u)
    clave = base.casefold()
    archivo = destino / base
    ocupado = usados.get(clave, u) != u
    if archivo.exists():
        registro = _ruta_origen(archivo)
        propietario = registro.read_text(encoding="utf-8").strip() if registro.is_file() else None
        ocupado = ocupado or propietario != u
    if ocupado:
        raise _ColisionDescarga(f"Colisión de nombres para {base}; usa otro --destino para este producto.")
    usados[clave] = u
    return base


def _carpetas_descarga(urls, destino):
    """Si hay una colisión, mueve juntos los archivos de ese directorio remoto."""
    por_fuente = {}
    for u in urls:
        try:
            partes = urlsplit(u)
        except ValueError:
            continue  # el bucle de descarga informa el error de esta URL
        directorio = partes.path.rpartition("/")[0]
        fuente = partes._replace(path=directorio, query="", fragment="").geturl()
        por_fuente.setdefault(fuente, []).append(u)

    carpetas = {}
    reservados = {}
    for fuente, grupo in por_fuente.items():
        huella = hashlib.sha256(fuente.encode()).hexdigest()[:12]
        alterna = destino / f"fuente_{huella}"
        if not alterna.is_dir():
            prueba = reservados.copy()
            for u in grupo:
                try:
                    _nombre_local(u, destino, prueba)
                except _ColisionDescarga:
                    print(f"Se guardarán juntos los archivos de {fuente} en {alterna}")
                    break
                except (ValueError, OSError):
                    continue  # el descargador informa errores por archivo
            else:
                reservados = prueba
                continue
        for u in grupo:
            carpetas[u] = alterna
    return carpetas


def _descargar_uno(requests, u, f, reintentos):
    """Descarga u -> f con reanudación (Range), verificación de tamaño y reintentos."""
    tmp = f.with_suffix(f.suffix + ".part")
    meta_path = tmp.with_name(tmp.name + ".json")
    if reintentos < 1:
        raise ValueError("--reintentos debe ser positivo.")
    err = None
    for intento in range(1, reintentos + 1):
        hecho = tmp.stat().st_size if tmp.exists() else 0
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
            if not isinstance(meta, dict):
                meta = {}
        except (ValueError, OSError):
            meta = {}
        if hecho and (meta.get("url") != u or not meta.get("validator")):
            tmp.unlink()
            hecho = 0
        cab = {"Accept-Encoding": "identity"}
        if hecho:
            cab.update(Range=f"bytes={hecho}-", **{"If-Range": meta["validator"]})
        try:
            with requests.get(u, stream=True, timeout=60, headers=cab) as r:
                if r.status_code == 416:  # rango no satisfacible
                    total_srv = re.fullmatch(r"bytes \*/(\d+)", r.headers.get("Content-Range", ""))
                    val = r.headers.get(meta.get("validator_header", "ETag"))
                    if (hecho and total_srv and int(total_srv[1]) == hecho == meta.get("total")
                            and val == meta.get("validator")):
                        break  # el .part ya estaba completo
                    tmp.unlink(missing_ok=True)
                    meta_path.unlink(missing_ok=True)
                    raise OSError("el servidor rechazó el rango; se reinicia la descarga")
                r.raise_for_status()
                if r.status_code not in (200, 206):
                    raise OSError(f"Respuesta inesperada para un archivo: HTTP {r.status_code}")
                if r.headers.get("Content-Encoding", "identity").lower() != "identity":
                    raise OSError("El servidor envió datos comprimidos pese a Accept-Encoding: identity.")
                reanuda = hecho > 0 and r.status_code == 206
                clen_text = r.headers.get("Content-Length")
                if clen_text is not None and not re.fullmatch(r"\d+", clen_text):
                    raise OSError("Content-Length inválido.")
                clen = int(clen_text) if clen_text is not None else None
                if r.status_code == 206:
                    rango = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", r.headers.get("Content-Range", ""))
                    if not rango:
                        raise OSError("Falta un Content-Range completo y válido.")
                    inicio, fin, esperado = map(int, rango.groups())
                    val = r.headers.get(meta.get("validator_header", "ETag"))
                    if (inicio != hecho or not inicio <= fin < esperado
                            or (clen is not None and clen != fin - inicio + 1)
                            or (reanuda and val is not None and val != meta.get("validator"))
                            or (reanuda and meta.get("total") not in (None, esperado))):
                        tmp.unlink(missing_ok=True)
                        meta_path.unlink(missing_ok=True)
                        raise OSError("Content-Range o versión no coincide; se reinicia la descarga.")
                    fin_respuesta = fin + 1
                else:
                    hecho = 0  # respuesta completa; nunca añadir a una parte anterior
                    esperado = clen
                    fin_respuesta = esperado
                etag = r.headers.get("ETag", "")
                cab_val = "ETag" if etag and not etag.startswith("W/") else "Last-Modified"
                val = r.headers.get(cab_val)
                if reanuda and not val:
                    cab_val, val = meta["validator_header"], meta["validator"]
                _escribir_texto(meta_path, json.dumps({"url": u, "validator": val,
                    "validator_header": cab_val, "total": esperado}))
                with open(tmp, "ab" if reanuda else "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        if not chunk:
                            continue
                        if fin_respuesta is not None and hecho + len(chunk) > fin_respuesta:
                            raise OSError("La respuesta contiene más bytes que los declarados.")
                        fh.write(chunk)
                        hecho += len(chunk)
                        msg = f"\r  {hecho / 1e6:8.1f} MB"
                        if esperado:
                            msg += f" / {esperado / 1e6:.1f} MB"
                        print(msg, end="", flush=True)
            actual = tmp.stat().st_size
            if actual == 0 or (esperado is not None and actual != esperado):
                raise OSError(f"descarga incompleta ({actual} de {esperado} bytes)")
            break
        except requests.HTTPError as e:
            code = e.response.status_code if e.response is not None else 0
            if 400 <= code < 500 and code not in (408, 429):
                raise  # 404, 403... no tiene sentido reintentar
            err = e
        except (requests.RequestException, OSError) as e:
            err = e
        print(f"\n  intento {intento}/{reintentos} falló: {err}")
        if intento < reintentos:
            time.sleep(min(60, 2 ** intento))
    else:
        raise OSError(f"No se pudo descargar {u}: {err}")
    tmp.replace(f)  # replace (no rename): sobrescribe también en Windows
    meta_path.unlink(missing_ok=True)


def _descargar_urls(requests, urls, destino, reintentos, forzar):
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    _migrar_registros_url(destino)
    urls = list(dict.fromkeys(urls))
    print(f"Descargando {len(urls)} archivo(s) en {destino}")
    carpetas = _carpetas_descarga(urls, destino)
    usados_por_carpeta, fallos = {}, []
    for u in urls:
        carpeta = carpetas.get(u, destino)
        try:
            carpeta.mkdir(parents=True, exist_ok=True)
            usados = usados_por_carpeta.setdefault(carpeta, {})
            f = carpeta / _nombre_local(u, carpeta, usados)
            if f.exists() and not forzar:
                print("Ya existe, se omite:", f)
                continue
            print("Descargando", u)
            _descargar_uno(requests, u, f, reintentos)
            _escribir_texto(_ruta_origen(f), u)
        except (requests.RequestException, OSError, ValueError) as e:
            print("\n  ERROR:", e)
            fallos.append(u)
            continue
        print("\n  guardado:", f)
    print("Recuerda: en PDS3 el .lbl y el .img deben estar en la misma carpeta.")
    if fallos:
        sys.exit(f"{len(fallos)} descarga(s) fallaron:\n  " + "\n  ".join(fallos))


def cmd_descargar(args):
    urls = list(args.url or [])
    if args.lista:
        urls += [l.strip() for l in Path(args.lista).read_text(encoding="utf-8-sig").splitlines()
                 if l.strip() and not l.lstrip().startswith("#")]
    if not urls:
        sys.exit("Indica --url o --lista")
    _descargar_urls(_importar("requests"), urls, args.destino, args.reintentos, args.forzar)


# --------------------------------------------------------------------------
# 3) inspeccionar
# --------------------------------------------------------------------------
def inspeccionar(archivo):
    rasterio, Resampling = _importar_rasterio()
    with rasterio.open(archivo) as src:
        if src.count < 1:
            raise ValueError("El raster no contiene bandas.")
        print("Driver        :", src.driver)
        print("Tamaño (ancho x alto):", src.width, "x", src.height)
        print("Bandas / dtype:", src.count, "/", src.dtypes[0])
        print("CRS           :", src.crs)
        print("Transform     :", src.transform)
        print("Límites       :", src.bounds)
        print("Res (unid. CRS):", src.res)
        print("Nodata        :", src.nodata)
        print("Escalas/Offsets:", src.scales, "/", src.offsets)
        print("Unidades      :", src.units)
        print("Tags (10)     :", dict(list(src.tags().items())[:10]))
        if src.crs and src.crs.is_geographic and src.bounds.left >= 0 and src.bounds.right > 180:
            print("AVISO: longitudes en convención 0-360 (no -180..180).")
        f = max(1, math.ceil(max(src.width, src.height) / 1000))
        a = src.read(1, out_shape=(max(1, src.height // f), max(1, src.width // f)),
                     resampling=Resampling.nearest, masked=True)
        v = a.compressed()
        v = v[np.isfinite(v)]
        if v.size:
            print(f"Estadísticas (muestra 1/{f}): min={v.min():.3f} max={v.max():.3f} media={v.mean():.3f}")
            print("  (valores crudos; aplica escala/offset si no son 1/0)")
        else:
            print("Toda la muestra está enmascarada (nodata). Revisa nodata/escala.")


def cmd_inspeccionar(args):
    inspeccionar(args.archivo)


# --------------------------------------------------------------------------
# 4) procesar: remuestreo, pendiente, rugosidad
# --------------------------------------------------------------------------
def _metricas_filas(transform, crs, filas, fila0=0):
    """Espaciado real por fila para grados y la proyección cilíndrica MOLA."""
    radio, e2 = _elipsoide(crs)
    parametros = crs.to_dict()
    y = transform.f + (np.arange(fila0, fila0 + filas) + .5) * transform.e
    if crs.is_geographic:
        lat = y
        dlon, dlat = np.radians([abs(transform.a), abs(transform.e)])
    elif parametros.get("proj") == "eqc":
        from rasterio.warp import transform as transformar
        # Invertir el propio CRS conserva su radio y origen, incluidos archivos PDS.
        _, lat = transformar(crs, _crs_geografico(crs), np.full(filas, transform.c), y)
        lat = np.asarray(lat)
        k = _factor_lineal(crs)
        dlon = abs(transform.a) * k / (radio * math.cos(math.radians(parametros.get("lat_ts", 0))))
        dlat = abs(transform.e) * k / radio
    else:
        return None
    sen2 = np.sin(np.radians(lat)) ** 2
    normal = radio / np.sqrt(1 - e2 * sen2)
    meridiano = radio * (1 - e2) / (1 - e2 * sen2) ** 1.5
    return normal * np.cos(np.radians(lat)) * dlon, meridiano * dlat, lat


def _gradiente_metrico(z, transform, crs, fila0, columna0, lat_max):
    """Convierte diferencias por píxel a derivadas sobre la superficie del elipsoide."""
    dz_fila, dz_col = np.gradient(z)
    metrica = _metricas_filas(transform, crs, z.shape[0], fila0)
    if metrica is not None:
        dx, dy, lat = metrica
        valido = (np.abs(lat) <= lat_max) & (np.abs(lat) < 90) & (dx > 0) & (dy > 0)
        dx = np.where(valido, dx, np.nan)
        dy = np.where(valido, dy, np.nan)
        return dz_col / dx[:, None], dz_fila / dy[:, None]

    # Otras proyecciones (incluidas polares): Jacobiano local de la proyección
    # inversa. Resolver ambas direcciones también cubre grillas no ortogonales
    # sobre el terreno. No se confunden metros de mapa con metros de superficie.
    from rasterio.warp import transform as transformar
    radio, e2 = _elipsoide(crs)
    yy, xx = np.meshgrid(
        transform.f + (np.arange(fila0, fila0 + z.shape[0]) + .5) * transform.e,
        transform.c + (np.arange(columna0, columna0 + z.shape[1]) + .5) * transform.a,
        indexing="ij")
    geografico = _crs_geografico(crs)

    def invertir(x, y):
        lon, lat = transformar(crs, geografico, x.ravel(), y.ravel())
        return np.asarray(lon).reshape(z.shape), np.asarray(lat).reshape(z.shape)

    _, lat = invertir(xx, yy)
    phi = np.radians(lat)
    normal = radio / np.sqrt(1 - e2 * np.sin(phi) ** 2)
    meridiano = radio * (1 - e2) / (1 - e2 * np.sin(phi) ** 2) ** 1.5
    delta = .001  # fracción de píxel; derivada local, no distancia entre centros

    def base(dx, dy):
        lon1, lat1 = invertir(xx - dx * delta, yy - dy * delta)
        lon2, lat2 = invertir(xx + dx * delta, yy + dy * delta)
        dlon = np.radians((lon2 - lon1 + 180) % 360 - 180) / (2 * delta)
        dlat = np.radians(lat2 - lat1) / (2 * delta)
        return normal * np.cos(phi) * dlon, meridiano * dlat

    ex, nx = base(transform.a, 0)
    ey, ny = base(0, transform.e)
    det = ex * ny - ey * nx
    valido = (np.abs(lat) <= lat_max) & (np.abs(lat) < 90) & np.isfinite(det) & (np.abs(det) > 1e-12)
    det = np.where(valido, det, np.nan)
    return (dz_col * ny - dz_fila * nx) / det, (dz_fila * ex - dz_col * ey) / det


def pendiente_y_rugosidad(z, transform, crs, fila0=0, lat_max=88.0, tam_filtro=3, columna0=0):
    """
    z: elevación (m), NaN en vacíos (puede ser una banda de filas del raster completo).
    transform: geotransform del raster COMPLETO; fila0 = índice de la primera fila de z
    dentro de ese raster (necesario para la latitud correcta de cada fila).

    Pendiente (grados) por diferencias centrales; rugosidad = desviación estándar de la
    pendiente en una ventana tam_filtro x tam_filtro. Momentos en float64 para
    evitar pérdida de precisión y diferencias entre bloques; salida float32.
    """
    uniform_filter = _importar("scipy.ndimage", "scipy").uniform_filter

    _validar_grilla(transform, crs)
    if z.ndim != 2 or min(z.shape) < 2:
        raise ValueError("El cálculo necesita al menos 2x2 celdas.")
    if not isinstance(tam_filtro, int) or tam_filtro < 3 or tam_filtro % 2 == 0:
        raise ValueError("--tam-filtro debe ser un entero impar >= 3.")
    if not math.isfinite(lat_max) or not 0 < lat_max <= 90:
        raise ValueError("--lat-max debe estar entre 0 (excluido) y 90.")
    z = np.ma.asarray(z, dtype="float64").filled(np.nan)
    z = np.where(np.isfinite(z), z, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        dzdx, dzdy = _gradiente_metrico(z, transform, crs, fila0, columna0, lat_max)
        pend = np.hypot(dzdx, dzdy)
    del dzdx, dzdy
    np.arctan(pend, out=pend)
    np.degrees(pend, out=pend)
    pend[~np.isfinite(z)] = np.nan

    valido = np.isfinite(pend)
    c = np.where(valido, pend, 0.)
    n = uniform_filter(valido.astype("float64"), size=tam_filtro)
    m1 = uniform_filter(c, size=tam_filtro)
    m2 = uniform_filter(c * c, size=tam_filtro)
    with np.errstate(invalid="ignore", divide="ignore"):
        m1 /= n
        m2 /= n
        m2 -= m1 * m1
        np.clip(m2, 0, None, out=m2)
        rug = np.sqrt(m2)
    rug = np.where((n < 0.5) | ~valido, np.nan, rug).astype("float32", copy=False)
    return pend.astype("float32", copy=False), rug


def _normalizar_bbox(bbox, src):
    """
    Valida el bbox y lo lleva a la convención de longitud del archivo (0-360 o -180..180).
    Un bbox que cruza el meridiano de corte del archivo no se admite (haría falta un mosaico).
    """
    lonmin, lonmax, latmin, latmax = bbox
    if not all(math.isfinite(v) for v in bbox) or lonmin >= lonmax or latmin >= latmax:
        raise ValueError("bbox inválido: valores finitos, MIN < MAX en ambos ejes.")
    if src.crs is None or not src.crs.is_geographic:
        return lonmin, lonmax, latmin, latmax  # CRS proyectado: el bbox va en unidades del CRS
    if latmin < -90 or latmax > 90 or lonmax - lonmin > 360 or lonmin < -180 or lonmax > 360:
        raise ValueError("bbox geográfico fuera de los rangos de latitud/longitud admitidos.")
    if src.bounds.left >= 0 and src.bounds.right > 180:  # archivo en 0-360
        if lonmin < 0 < lonmax:
            raise ValueError("El bbox cruza el meridiano 0 y el archivo usa longitudes 0-360. "
                     "Procesa dos recortes (p. ej. 0..LONMAX y 360+LONMIN..360) por separado.")
        if lonmax <= 0:
            lonmin, lonmax = lonmin + 360.0, lonmax + 360.0
    elif src.bounds.right <= 180 and lonmax > 180:  # archivo en -180..180
        if lonmin < 180:
            raise ValueError("El bbox cruza el antimeridiano (180°) y el archivo usa -180..180. "
                     "Procesa dos recortes por separado.")
        lonmin, lonmax = lonmin - 360.0, lonmax - 360.0
    return lonmin, lonmax, latmin, latmax


def procesar(archivo, salida, res_km, bbox=None, lat_max=88.0, filas_banda=256,
             tam_filtro=3, columnas_bloque=1024, cuerpo="marte", unidad_z=None):
    """Remuestrea y calcula por ventanas con halo, sin cargar el DEM completo.

    Conserva el CRS. res_km es la resolución nominal: en grados/cilíndrica el
    ancho físico disminuye hacia los polos. No crea detalle mediante sobremuestreo.
    bbox usa las coordenadas NATIVAS del archivo (grados o unidades proyectadas).
    """
    rasterio, Resampling = _importar_rasterio()
    from affine import Affine
    from rasterio.errors import WindowError
    from rasterio.windows import Window, from_bounds, intersection, transform as win_transform
    from rasterio.vrt import WarpedVRT

    if not math.isfinite(res_km) or res_km <= 0:
        raise ValueError("--res-km debe ser finito y positivo.")
    if not math.isfinite(lat_max) or not 0 < lat_max <= 90:
        raise ValueError("--lat-max debe estar entre 0 (excluido) y 90.")
    if not isinstance(tam_filtro, int) or tam_filtro < 3 or tam_filtro % 2 == 0:
        raise ValueError("--tam-filtro debe ser impar y >= 3.")
    if (not isinstance(filas_banda, int) or filas_banda < 1
            or not isinstance(columnas_bloque, int) or columnas_bloque < 1):
        raise ValueError("Los tamaños de bloque deben ser enteros positivos.")
    if Path(archivo).resolve() == Path(salida).resolve():
        raise ValueError("El archivo de salida no puede ser el archivo de entrada.")
    halo = 1 + tam_filtro // 2

    with rasterio.Env(GDAL_CACHEMAX=64 * 1024 * 1024), rasterio.open(archivo) as src:
        _validar_grilla(src.transform, src.crs, cuerpo)
        if src.count != 1:
            raise ValueError("Se requiere un DEM de una sola banda, no una salida ya procesada.")
        if np.issubdtype(np.dtype(src.dtypes[0]), np.complexfloating):
            raise ValueError("Un DEM no puede contener números complejos.")
        # Los otros MEGDR no representan elevaciones sobre el areoide.
        if re.match(r"(?i)^meg[acr](?:\d|_)", Path(archivo).stem):
            raise ValueError("Usa el producto MEGT de topografía, no radio, areoide o recuentos.")
        completa = Window(0, 0, src.width, src.height)
        if bbox is not None:
            lonmin, lonmax, latmin, latmax = _normalizar_bbox(bbox, src)
            v = from_bounds(lonmin, latmin, lonmax, latmax, transform=src.transform)
            # Incluir todas las celdas intersectadas, no redondear la longitud.
            def ajustar(x):
                n = round(x)
                return n if abs(x - n) < 1e-8 else x
            c0, r0 = math.floor(ajustar(v.col_off)), math.floor(ajustar(v.row_off))
            c1 = math.ceil(ajustar(v.col_off + v.width))
            r1 = math.ceil(ajustar(v.row_off + v.height))
            v = Window(c0, r0, c1 - c0, r1 - r0)
            try:
                ventana = intersection(v, completa)  # recorta a lo que realmente existe
            except WindowError:
                raise ValueError("El bbox no se solapa con el raster. Revisa la convención de longitud "
                         "(ver 'inspeccionar').")
        else:
            ventana = completa
        ventana = Window(int(ventana.col_off), int(ventana.row_off),
                         int(ventana.width), int(ventana.height))
        if ventana.width < 1 or ventana.height < 1:
            raise ValueError("El bbox queda vacío tras recortarlo al raster.")

        if src.crs is not None and src.crs.is_projected:
            paso = res_km * 1000.0 / _factor_lineal(src.crs)  # en unidades del CRS
        else:
            paso = res_km * 1000.0 / (_elipsoide(src.crs)[0] * math.pi / 180.0)

        out_w = max(1, min(ventana.width, int(round(ventana.width * abs(src.transform.a) / paso))))
        out_h = max(1, min(ventana.height, int(round(ventana.height * abs(src.transform.e) / paso))))
        if out_h < 3 or out_w < 3:
            raise ValueError(f"La región/resolución produce un raster de {out_w}x{out_h} px (mínimo 3x3). "
                     "Reduce --res-km o amplía --bbox.")

        esc = src.scales[0] if src.scales and src.scales[0] is not None else 1.0
        off = src.offsets[0] if src.offsets and src.offsets[0] is not None else 0.0
        if not math.isfinite(esc) or not math.isfinite(off) or esc == 0:
            raise ValueError("Escala/offset de elevación inválidos.")
        unidad = unidad_z or src.units[0] or "m"
        factor_z = _factor_vertical(unidad)

        base_t = win_transform(ventana, src.transform)
        t = Affine(base_t.a * ventana.width / out_w, 0, base_t.c,
                   0, base_t.e * ventana.height / out_h, base_t.f)
        crs = src.crs
        salida = Path(salida)
        salida.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=salida.parent, suffix=".tif", delete=False) as f:
            tmp = Path(f.name)
        try:
            # Una sola grilla virtual evita desplazamientos al remuestrear bloques
            # cuyo tamaño no es múltiplo entero de la resolución de entrada.
            with WarpedVRT(src, crs=crs, transform=t, width=out_w, height=out_h,
                           dtype="float32", nodata=np.nan, resampling=Resampling.average,
                           warp_mem_limit=32) as vrt, rasterio.open(
                    tmp, "w", driver="GTiff", height=out_h, width=out_w, count=3,
                    dtype="float32", crs=crs, transform=t, nodata=np.nan,
                    compress="deflate", predictor=3, tiled=True,
                    blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER") as dst:
                validos = [0, 0, 0]
                for r0 in range(0, out_h, filas_banda):
                    r1 = min(out_h, r0 + filas_banda)
                    a, b = max(0, r0 - halo), min(out_h, r1 + halo)
                    for c0 in range(0, out_w, columnas_bloque):
                        c1 = min(out_w, c0 + columnas_bloque)
                        ca, cb = max(0, c0 - halo), min(out_w, c1 + halo)
                        arr = vrt.read(1, window=Window(ca, a, cb-ca, b-a), masked=True)
                        z = arr.filled(np.nan).astype("float64")
                        z = ((z * esc + off) * factor_z).astype("float32")
                        z[~np.isfinite(z)] = np.nan
                        pend, rug = pendiente_y_rugosidad(z, t, crs, fila0=a, columna0=ca,
                                                       lat_max=lat_max, tam_filtro=tam_filtro)
                        sl = (slice(r0-a, r1-a), slice(c0-ca, c1-ca))
                        w = Window(c0, r0, c1-c0, r1-r0)
                        for i, banda in enumerate((z, pend, rug), 1):
                            dst.write(banda[sl], i, window=w)
                            validos[i-1] += int(np.isfinite(banda[sl]).sum())
                dst.set_band_description(1, "elevacion_m")
                dst.set_band_description(2, "pendiente_deg")
                dst.set_band_description(3, "rugosidad_sd_pendiente_deg")
                for i, unidad_out in enumerate(("m", "degree", "degree"), 1):
                    dst.set_band_unit(i, unidad_out)
                    dst.update_tags(i, valid_pixels=validos[i-1], total_pixels=out_w*out_h)
                dst.update_tags(
                    source=Path(archivo).name, body=cuerpo, requested_resolution_km=res_km,
                    resampling="average", roughness_window_pixels=tam_filtro,
                    roughness_definition="standard_deviation_of_slope_degrees",
                    roughness_min_valid_fraction=.5, latitude_limit=lat_max,
                    radius_semi_major_m=_elipsoide(crs)[0], vertical_input_unit=unidad,
                    vertical_unit_assumed=str(not unidad_z and not src.units[0]).lower(),
                    input_scale=esc, input_offset=off,
                    slope_method="central_differences_local_surface_metric",
                    resolution_note="Nominal grid spacing; physical footprint varies with latitude/projection.")
            tmp.replace(salida)
        finally:
            tmp.unlink(missing_ok=True)
    return out_h, out_w


def _leer_bandas(ruta, max_lado=2000):
    """Lee las 3 bandas de salida, submuestreando si son enormes (solo para estadísticas)."""
    rasterio, Resampling = _importar_rasterio()
    with rasterio.open(ruta) as ds:
        f = max(1, math.ceil(max(ds.width, ds.height) / max_lado))
        forma = (max(1, ds.height // f), max(1, ds.width // f))
        return [ds.read(i, out_shape=forma, resampling=Resampling.nearest) for i in (1, 2, 3)]


def _procesar_archivo(archivo, salida, args):
    inspeccionar(archivo)
    out_h, out_w = procesar(archivo, salida, args.res_km, args.bbox,
                            args.lat_max, args.filas_banda, args.tam_filtro,
                            args.columnas_bloque, args.cuerpo, args.unidad_z)
    print("Guardado:", salida, "| forma:", (out_h, out_w))
    z, pend, rug = _leer_bandas(salida)
    resumen = {"source": str(archivo), "output_shape": [out_h, out_w],
               "statistics": "regular pixel sample; not area-weighted", "sample_shape": list(z.shape),
               "bands": {}}
    for nombre, a in (("elevación (m)", z), ("pendiente (°)", pend), ("rugosidad (°)", rug)):
        v = a[np.isfinite(a)]
        if v.size:
            resumen["bands"][nombre] = dict(zip(("p5", "median", "p95"),
                                                     map(float, np.percentile(v, [5, 50, 95]))))
            print(f"  {nombre:14s} p5={np.percentile(v, 5):9.3f}  mediana={np.median(v):9.3f}  p95={np.percentile(v, 95):9.3f}")
        else:
            resumen["bands"][nombre] = None
    rasterio, _ = _importar_rasterio()
    with rasterio.open(salida) as ds:
        resumen.update(crs=ds.crs.to_wkt(), bounds=list(ds.bounds), grid_resolution=list(ds.res),
                       processing=ds.tags(), valid_pixels=[int(ds.tags(i)["valid_pixels"]) for i in (1, 2, 3)])
        metrica = _metricas_filas(ds.transform, ds.crs, ds.height)
        if metrica is not None:
            dx, dy, lat = metrica
            m = (np.abs(lat) <= args.lat_max) & (np.abs(lat) < 90)
            if m.any():
                resumen["pixel_size_m"] = {"x_min": float(dx[m].min()), "x_max": float(dx[m].max()),
                                           "y_min": float(dy[m].min()), "y_max": float(dy[m].max())}
                print(f"  Tamaño físico de celda: X={dx[m].min():.1f}..{dx[m].max():.1f} m; "
                      f"Y={dy[m].min():.1f}..{dy[m].max():.1f} m")
    ruta_resumen = Path(salida).with_suffix(".json")
    _escribir_texto(ruta_resumen, json.dumps(resumen, ensure_ascii=False, indent=2, allow_nan=False))
    print("Resumen:", ruta_resumen)
    print("AVISO: resolución nominal, sin sobremuestreo. La huella física depende de la latitud/proyección. "
          "Compara Tierra y Marte a igual escala física y ventana; estos derivados no evalúan habitabilidad.")


def _procesar_carpeta(entrada, destino, args):
    if not entrada.is_dir():
        raise ValueError(f"No se encontró la carpeta {entrada}. Usa 'buscar' para descargar los datos.")
    if not destino.is_dir() and (destino.exists() or destino.suffix.lower() in {".tif", ".tiff"}):
        raise ValueError("Al procesar una carpeta, --salida debe indicar un directorio.")
    etiquetas = sorted(
        (p for p in entrada.rglob("*") if p.suffix.lower() == ".lbl" and p.is_file()),
        key=lambda p: p.as_posix(),
    )
    if not etiquetas:
        raise ValueError(f"No hay etiquetas .lbl en {entrada} ni en sus subcarpetas.")

    _importar_rasterio()
    from rasterio.errors import RasterioError

    nombres = set()
    fallos = 0
    for i, etiqueta in enumerate(etiquetas, 1):
        relativa = etiqueta.relative_to(entrada)
        nombre = etiqueta.stem
        if relativa.parent != Path(".") or nombre.casefold() in nombres:
            # La ruta relativa distingue etiquetas con el mismo nombre en distintas carpetas.
            huella = hashlib.sha256(relativa.as_posix().encode()).hexdigest()[:12]
            nombre = f"{nombre}__{huella}"
        base = nombre
        numero = 2
        while nombre.casefold() in nombres:
            nombre = f"{base}__{numero}"
            numero += 1
        nombres.add(nombre.casefold())
        print(f"\n[{i}/{len(etiquetas)}] {etiqueta}", flush=True)
        try:
            _procesar_archivo(etiqueta, destino / f"{nombre}.tiff", args)
        except (OSError, ValueError, RasterioError) as e:
            print(f"Error al procesar {etiqueta}: {e}", file=sys.stderr)
            fallos += 1
    print(f"Etiquetas procesadas: {len(etiquetas) - fallos}/{len(etiquetas)}. Fallos: {fallos}.")
    if fallos:
        sys.exit(1)


def cmd_procesar(args):
    entrada = Path(args.archivo or DESTINO_CRUDO)
    if not args.archivo or entrada.is_dir():
        _procesar_carpeta(entrada, Path(args.salida or Path(DESTINO_CRUDO).parent), args)
    else:
        _procesar_archivo(args.archivo, args.salida or "datos_marte/mola_derivados.tif", args)


# --------------------------------------------------------------------------
# 5) selftest
# --------------------------------------------------------------------------
def cmd_selftest(_args):
    rasterio, _ = _importar_rasterio()
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    pendiente_real = 0.10  # 10 % de pendiente en x -> atan(0.1) = 5.71 grados
    esperado = math.degrees(math.atan(pendiente_real))
    res_deg = 1.0 / 16.0
    lat0, n = 2.0, 64
    res_km_nat = res_deg * (R_MARTE_M * math.pi / 180.0) / 1000.0
    fallos = []

    def check(nombre, cond, detalle=""):
        print(f"  {'OK   ' if cond else 'FALLO'} {nombre} {detalle}")
        if not cond:
            fallos.append(nombre)

    def rampa(ruta, lon0, escala=None, hueco=False):
        lon = (np.arange(n) + 0.5) * res_deg
        x_m = R_MARTE_M * np.radians(lon)
        z = np.tile((pendiente_real * x_m)[None, :], (n, 1))
        perfil = dict(driver="GTiff", height=n, width=n, count=1, dtype="float32",
                      crs=CRS.from_proj4(f"+proj=longlat +R={R_MARTE_M} +no_defs"),
                      transform=from_origin(lon0, lat0, res_deg, res_deg))
        if escala:
            esc, off = escala
            z = (z - off) / esc
        if hueco:
            z[20:28, 20:28] = -9999.0
            perfil["nodata"] = -9999.0
        with rasterio.open(ruta, "w", **perfil) as dst:
            dst.write(z.astype("float32"), 1)
            if escala:
                dst.scales = (escala[0],)
                dst.offsets = (escala[1],)

    def interior(a):
        return a[3:-3, 3:-3]

    def falla_con_exit(**kw):
        try:
            procesar(**kw)
        except (SystemExit, ValueError):
            return True
        return False

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        base = str(d / "rampa.tif")
        rampa(base, 0.0)

        print("1) resolución nativa")
        procesar(base, str(d / "o1.tif"), res_km_nat)
        _, p, r = _leer_bandas(d / "o1.tif")
        med, rug = float(np.nanmedian(interior(p))), float(np.nanmedian(interior(r)))
        check("pendiente", abs(med - esperado) < 0.1, f"(esperada {esperado:.3f}°, obtenida {med:.3f}°)")
        check("rugosidad ~0", rug < 0.05, f"({rug:.4f}°)")

        print("2) remuestreo a la mitad de resolución (camino 'average' + geotransform)")
        procesar(base, str(d / "o2.tif"), res_km_nat * 2)
        _, p, _ = _leer_bandas(d / "o2.tif")
        check("forma 32x32", p.shape == (32, 32), f"(forma {p.shape})")
        med = float(np.nanmedian(interior(p)))
        check("pendiente", abs(med - esperado) < 0.1, f"({med:.3f}°)")

        print("3) bbox")
        h, w = procesar(base, str(d / "o3.tif"), res_km_nat, bbox=(1.0, 3.0, -1.0, 1.0))
        _, p, _ = _leer_bandas(d / "o3.tif")
        check("forma 32x32", (h, w) == (32, 32), f"({(h, w)})")
        med = float(np.nanmedian(interior(p)))
        check("pendiente", abs(med - esperado) < 0.1, f"({med:.3f}°)")

        print("4) bbox parcialmente fuera del raster (se recorta)")
        h, w = procesar(base, str(d / "o4.tif"), res_km_nat, bbox=(3.0, 10.0, -1.0, 1.0))
        check("forma recortada", w == 16 and h == 32, f"({(h, w)})")
        check("bbox disjunto -> error",
              falla_con_exit(archivo=base, salida=str(d / "x.tif"), res_km=res_km_nat,
                             bbox=(50.0, 60.0, -1.0, 1.0)))

        print("5) escala y offset")
        esc_off = str(d / "rampa_so.tif")
        rampa(esc_off, 0.0, escala=(0.5, 100.0))
        procesar(esc_off, str(d / "o5.tif"), res_km_nat)
        z, p, _ = _leer_bandas(d / "o5.tif")
        med = float(np.nanmedian(interior(p)))
        check("pendiente", abs(med - esperado) < 0.1, f"({med:.3f}°)")
        check("elevación con escala/offset aplicados", abs(float(z[32, 0]) - float(z[32, 1])) > 100)

        print("6) hueco de nodata")
        hueco = str(d / "rampa_hueco.tif")
        rampa(hueco, 0.0, hueco=True)
        procesar(hueco, str(d / "o6.tif"), res_km_nat)
        _, p, r = _leer_bandas(d / "o6.tif")
        check("centro del hueco enmascarado", np.isnan(p[24, 24]) and np.isnan(r[24, 24]))
        med = float(np.nanmedian(p))
        check("pendiente fuera del hueco", abs(med - esperado) < 0.1, f"({med:.3f}°)")
        check("sin NaN lejos del hueco", np.isfinite(p[50, 50]) and np.isfinite(r[50, 50]))

        print("7) por bandas == de una vez")
        procesar(base, str(d / "o7a.tif"), res_km_nat, filas_banda=8)
        _, pa, ra = _leer_bandas(d / "o7a.tif")
        _, pb, rb = _leer_bandas(d / "o1.tif")
        check("pendiente idéntica", np.allclose(pa, pb, atol=1e-3, equal_nan=True))
        check("rugosidad idéntica", np.allclose(ra, rb, atol=5e-3, equal_nan=True))

        print("8) raster demasiado pequeño")
        check("<3 px -> error",
              falla_con_exit(archivo=base, salida=str(d / "x.tif"), res_km=res_km_nat,
                             bbox=(1.0, 1.1, -0.05, 0.05)))

        print("9) longitudes 0-360")
        g360 = str(d / "rampa360.tif")
        rampa(g360, 350.0)  # cubre 350..354 -> convención 0-360
        h, w = procesar(g360, str(d / "o9.tif"), res_km_nat, bbox=(-8.0, -6.0, -1.0, 1.0))
        check("bbox negativo se convierte a 0-360", (h, w) == (32, 32), f"({(h, w)})")
        check("bbox que cruza el meridiano 0 -> error claro",
              falla_con_exit(archivo=g360, salida=str(d / "x.tif"), res_km=res_km_nat,
                             bbox=(-1.0, 1.0, -1.0, 1.0)))

    print("SELFTEST:", "OK" if not fallos else "FALLÓ -> " + ", ".join(fallos))
    sys.exit(0 if not fallos else 1)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _opciones_descarga(parser):
    parser.add_argument("--destino", default=DESTINO_CRUDO, help="directorio de descarga (predeterminado: %(default)s)")
    parser.add_argument("--forzar", action="store_true", help="vuelve a descargar archivos de la misma URL")
    parser.add_argument("--reintentos", type=_entero_positivo, default=5, help="intentos por archivo (reanuda con Range)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("buscar", help="Busca en ODE y descarga los archivos encontrados")
    b.add_argument("--endpoint", default=ODE_ENDPOINT)
    b.add_argument("--target", default="mars")
    b.add_argument("--ihid", default="MGS", help="host del instrumento")
    b.add_argument("--iid", default="MOLA", help="instrumento")
    b.add_argument("--pt", default="MEGDR", help="tipo de producto")
    b.add_argument("--results", default="cpf", help="campos ODE (debe incluir f para archivos)")
    b.add_argument("--limit", type=_entero_positivo, default=100, help="productos por página")
    b.add_argument("--offset", type=int, default=0, help="índice inicial, desde 0")
    b.add_argument("--paginas", type=_entero_positivo, default=1, help="máximo de páginas a consultar")
    b.add_argument("--pdsid", help="ID/patrón de producto (por defecto MEGT* para MEGDR)")
    b.add_argument("--todos-tipos", action="store_true", help="incluir radio, areoide y recuentos MEGDR")
    b.add_argument("--extra", nargs="*", help="parámetros extra clave=valor")
    b.add_argument("--filtro", help="subcadena para filtrar las URLs (p. ej. un nombre de archivo)")
    b.add_argument("--salida", default="datos_marte/urls_mola.txt")
    b.add_argument("--solo-lista", action="store_true", help="guarda las URLs sin descargar los archivos")
    _opciones_descarga(b)
    b.set_defaults(func=cmd_buscar)

    d = sub.add_parser("descargar", help="Descarga archivos por URL")
    d.add_argument("--url", nargs="*")
    d.add_argument("--lista", help="archivo de texto con una URL por línea")
    _opciones_descarga(d)
    d.set_defaults(func=cmd_descargar)

    i = sub.add_parser("inspeccionar", help="Metadatos y estadísticas de un raster")
    i.add_argument("archivo", help="ruta al .lbl (PDS3) o .tif")
    i.set_defaults(func=cmd_inspeccionar)

    pr = sub.add_parser("procesar", help="Inspecciona y procesa un DEM o todas las etiquetas .lbl de una carpeta")
    pr.add_argument("archivo", nargs="?",
                    help=f"archivo DEM o carpeta de entrada (por defecto: {DESTINO_CRUDO})")
    pr.add_argument("--res-km", type=_real_positivo, default=5.0, help="resolución nominal (km/píxel); no sobremuestrea")
    pr.add_argument("--bbox", type=float, nargs=4, metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
                    help="recorte en el CRS NATIVO: grados si geográfico, unidades del mapa si proyectado")
    pr.add_argument("--lat-max", type=float, default=88.0, help="se enmascara la pendiente más allá de esta latitud")
    pr.add_argument("--filas-banda", type=_entero_positivo, default=256,
                    help="filas por bloque (menos = menos RAM)")
    pr.add_argument("--columnas-bloque", type=_entero_positivo, default=1024,
                    help="columnas por bloque; limita RAM incluso en mapas globales")
    pr.add_argument("--tam-filtro", type=_entero_positivo, default=3, help="ventana impar de rugosidad >= 3")
    pr.add_argument("--cuerpo", choices=("marte", "tierra"), default="marte",
                    help="valida el cuerpo del CRS; usa tierra al preparar análogos terrestres")
    pr.add_argument("--unidad-z", choices=("m", "km", "ft", "us_survey_foot"),
                    help="unidad vertical explícita; por defecto metadatos, o metros si faltan")
    pr.add_argument("--salida", help="GeoTIFF para un archivo o carpeta para un lote (por defecto: datos_marte)")
    pr.set_defaults(func=cmd_procesar)

    s = sub.add_parser("selftest", help="Prueba con datos sintéticos")
    s.set_defaults(func=cmd_selftest)

    args = p.parse_args()
    try:
        args.func(args)
    except (ValueError, OSError) as e:
        p.exit(1, f"Error: {e}\n")


if __name__ == "__main__":
    main()
