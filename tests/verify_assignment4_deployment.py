"""Build-image HTTP verification; uses a temporary localhost-only container."""
import argparse
import http.client
import json
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', default='gentext-app:assignment4')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--examples', type=int, default=50000)
    parser.add_argument('--report', default='verification.json')
    args = parser.parse_args()
    name = f'gentext-assignment4-test-{int(time.time())}'
    subprocess.run(['docker','run','--rm','-d','--name',name,
                    '-p','127.0.0.1::80',args.image],check=True)
    try:
        binding = subprocess.check_output(['docker','port',name,'80/tcp'],text=True).strip()
        base = 'http://' + binding
        for attempt in range(120):
            try:
                with urllib.request.urlopen(base+'/',timeout=2) as response:
                    assert response.status == 200
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.HTTPException):
                time.sleep(1)
        else:
            logs = subprocess.check_output(['docker','logs',name],stderr=subprocess.STDOUT,text=True)
            raise RuntimeError('Container startup failed: ' + logs)
        def post(route, body):
            request = urllib.request.Request(base+route,data=json.dumps(body).encode(),
                                            headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=120) as response:
                return response.status,response.headers.get('Content-Type'),response.read()
        for kind in ['diffusion','energy']:
            route = '/generate_' + kind
            status,mime,png = post(route,{'count':4,'steps':20,'seed':42})
            assert status == 200 and mime == 'image/png'
            assert png.startswith(b'\x89PNG\r\n\x1a\n')
            assert png == post(route,{'count':4,'steps':20,'seed':42})[2]
            (ROOT/'assignment4_results'/f'{kind}_api_sample.png').write_bytes(png)
            for body in [{'count':0},{'count':9},{'steps':201},{'seed':-1}]:
                try:
                    post(route,body)
                except urllib.error.HTTPError as error:
                    assert error.code == 422
                else:
                    raise AssertionError('Invalid input accepted')
        for route,body in [('/generate',{'start_word':'the','length':5}),
                           ('/embedding',{'word':'hello'}),
                           ('/generate_with_rnn',{'start_word':'the','length':5}),
                           ('/generate_digits',{'count':4,'seed':42})]:
            assert post(route,body)[0] == 200
        # Reuse a valid PNG to exercise the pre-existing classifier endpoint.
        boundary='assignment4-check'
        png=(ROOT/'assignment4_results'/'diffusion_api_sample.png').read_bytes()
        payload=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="test.png"\r\nContent-Type: image/png\r\n\r\n'.encode()
                 +png+f'\r\n--{boundary}--\r\n'.encode())
        request=urllib.request.Request(base+'/classify',data=payload,
            headers={'Content-Type':f'multipart/form-data; boundary={boundary}'})
        with urllib.request.urlopen(request,timeout=120) as response:
            assert response.status==200
        script='''import json,torch
from pathlib import Path
report={"torch":torch.__version__,"cuda":torch.version.cuda,"checkpoints":{}}
for kind in ["diffusion","energy"]:
 c=torch.load(Path("/code/models")/f"cifar10_{kind}.pth",map_location="cpu",weights_only=True)
 report["checkpoints"][kind]={k:c[k] for k in ["epoch","examples","dataset","image_size","channels"]}
print(json.dumps(report))
'''
        details=json.loads(subprocess.check_output(['docker','exec',name,
            '/code/.venv/bin/python','-c',script],text=True))
        for checkpoint in details['checkpoints'].values():
            assert checkpoint['epoch']==args.epochs,checkpoint
            assert checkpoint['examples']==args.examples,checkpoint
        report={'http_endpoints':'passed','png_outputs':'passed','deterministic_seeds':'passed',
                'invalid_input_422':'passed','existing_api_routes':'passed',
                'runtime':details,'image':subprocess.check_output(
                    ['docker','image','inspect',args.image,'--format','{{.Id}}'],text=True).strip()}
        (ROOT/'assignment4_results'/args.report).write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
    finally:
        subprocess.run(['docker','stop',name],check=False,stdout=subprocess.DEVNULL)


if __name__=='__main__':
    main()
