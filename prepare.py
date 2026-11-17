"""Download attributed CC BY 4.0 UCI data; retain local files only."""
import hashlib,json,urllib.request,zipfile
from recflow import PRIVATE
URL='https://archive.ics.uci.edu/static/public/352/online+retail.zip'

def prepare():
    from openpyxl import load_workbook
    PRIVATE.mkdir(exist_ok=True)
    archive=PRIVATE/'online-retail.zip'
    if not archive.exists():
        partial=archive.with_suffix('.part')
        with urllib.request.urlopen(URL,timeout=120) as response,partial.open('wb') as output:
            while block:=response.read(1024*1024):output.write(block)
        partial.replace(archive)
    xlsx=PRIVATE/'retail.xlsx'
    with zipfile.ZipFile(archive) as z:
        name=next(n for n in z.namelist() if n.endswith('.xlsx'))
        with z.open(name) as source,xlsx.open('wb') as target:
            import shutil
            shutil.copyfileobj(source,target)
    wb=load_workbook(xlsx,read_only=True,data_only=True)
    events=[];total=0;duplicates=set()
    for invoice,item,title,qty,date,price,customer,*_ in wb.active.iter_rows(min_row=2,values_only=True):
        total+=1
        if customer is None or date is None or not title or not qty or qty<=0 or not price or price<=0 or str(invoice).upper().startswith('C'):continue
        if not str(item).strip().isdigit():continue
        # Stable pseudonym; not a claim of irreversible anonymization.
        user='buyer-'+hashlib.sha256(('recflow-demo:'+str(int(customer))).encode()).hexdigest()[:12]
        key=(str(invoice),str(item),user)
        if key in duplicates:continue
        duplicates.add(key)
        events.append(dict(user=user,item=str(item),title=str(title).strip(),date=date.isoformat()))
    wb.close()
    events.sort(key=lambda e:(e['date'],e['user'],e['item']))
    (PRIVATE/'events.json').write_text(json.dumps(events),encoding='utf-8')
    sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    (PRIVATE/'data-source.json').write_text(json.dumps(dict(url=URL,archive_sha256=sha,
        rows_read=total,retained_purchase_lines=len(events),license='CC BY 4.0',
        citation='Chen, D. (2015). Online Retail. UCI. DOI:10.24432/C5BW33',
        demographic_features_used=False),indent=2))
    print('Prepared',len(events),'purchase lines',flush=True)
if __name__=='__main__':prepare()
