"""Read fixed public VerSe members without downloading the full ZIP archive."""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import urllib.request
import zipfile

SOURCE='https://s3.bonescreen.de/public/VerSe-complete/dataset-verse19test.zip'


class RangeReader(io.RawIOBase):
    def __init__(self,url):
        if url!=SOURCE:raise ValueError('Only the declared public VerSe source is accepted')
        self.url=url;self.position=0
        with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=60) as response:
            self.length=int(response.headers['Content-Length'])
            self.etag=response.headers.get('ETag')
        if not self.etag or self.etag.startswith('W/') or self.length<=0:raise ValueError('Strong archive object identity required')
        self.request_count=0
        self.cache={}

    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.position

    def seek(self,offset,whence=0):
        position=offset if whence==0 else self.position+offset if whence==1 else self.length+offset
        if position<0:raise ValueError('Negative archive offset')
        self.position=position;return position

    def read(self,size=-1):
        size=self.length-self.position if size<0 else min(size,self.length-self.position)
        if size<0 or size>200*2**20:raise ValueError('Archive read exceeds selected-member limit')
        if not size:return b''
        start=self.position;end=start+size-1;key=(start,end)
        if key in self.cache:data=self.cache[key]
        else:
            request=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{end}','If-Match':self.etag})
            with urllib.request.urlopen(request,timeout=180) as response:
                if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{self.length}':
                    raise ValueError('Server did not honor the bounded archive range')
                if response.headers.get('ETag')!=self.etag:raise ValueError('Archive object changed between range requests')
                data=response.read(size+1)
            self.request_count+=1
            if len(data)!=size:raise ValueError('Truncated archive range')
            if size<2**20:self.cache[key]=data
        self.position+=size;return data


def download_cases(selection_path,output):
    selection=json.loads(Path(selection_path).read_text())
    if selection['source_url']!=SOURCE or len(selection['cases'])!=6:raise ValueError('Expected frozen six-case VerSe selection')
    output=Path(output);output.mkdir(parents=True,exist_ok=True);records=[]
    reader=RangeReader(SOURCE)
    with zipfile.ZipFile(reader) as archive:
        scans=sorted(n for n in archive.namelist() if re.fullmatch(r'dataset-verse19test/rawdata/sub-verse\d+/sub-verse\d+_ct.nii.gz',n))
        if [c['ct_member'] for c in selection['cases']]!=scans[:6]:raise ValueError('Selection differs from fixed first-six rule')
        for case in selection['cases']:
            subject=case['subject']
            if not re.fullmatch(r'sub-verse\d+',subject):raise ValueError('Invalid public subject identifier')
            if case['ct_member']!=f'dataset-verse19test/rawdata/{subject}/{subject}_ct.nii.gz' or case['centroid_member']!=f'dataset-verse19test/derivatives/{subject}/{subject}_seg-subreg_ctd.json':
                raise ValueError('CT and annotation subject identity mismatch')
            if not re.fullmatch(r'[0-9a-f]{64}',case.get('ct_sha256','')):raise ValueError('Expected pinned CT SHA256 before acquisition')
            record={'subject':case['subject']}
            for kind,key in [('ct','ct_member'),('centroids','centroid_member')]:
                member=case[key];name=PurePosixPath(member).name
                if '..' in PurePosixPath(member).parts or PurePosixPath(member).is_absolute():raise ValueError('Unsafe archive member')
                info=archive.getinfo(member)
                if info.file_size>200*2**20:raise ValueError('Oversized selected input')
                if kind=='ct' and info.file_size!=case['ct_bytes']:raise ValueError('Selected CT size changed')
                data=archive.read(member)  # ZIP CRC is verified by the standard library.
                digest=hashlib.sha256(data).hexdigest()
                if kind=='ct' and digest!=case['ct_sha256']:raise ValueError('Pinned CT source digest mismatch')
                if kind=='centroids' and digest!=case['centroid_sha256']:raise ValueError('Centroid source changed')
                path=output/name
                if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Existing source conflicts')
                path.write_bytes(data);record[kind+'_sha256']=digest;record[kind+'_file']=name
            record['archive_etag']=reader.etag;record['cumulative_range_requests']=reader.request_count
            records.append(record);print('VERSE_INPUT_VERIFIED',case['subject'],flush=True)
    (output/'source-manifest.json').write_text(json.dumps(records,indent=2))
    return records
