"""Exercise the real saved model via in-process HTTP; not a network load test."""
from fastapi.testclient import TestClient
from app import app

def main():
    with TestClient(app) as client:
        assert client.get('/health').status_code==200
        assert 'RecFlow' in client.get('/').text
        user=client.get('/users').json()['users'][0]
        result=client.get('/recommend',params={'user':user,'method':'collaborative'})
        assert result.status_code==200,result.text
        assert result.json()['recommendations']
        assert not result.json()['score_is_probability']
        assert client.get('/recommend',params={'user':'new-buyer'}).json()['strategy']=='popularity fallback'
        assert client.get('/recommend',params={'k':51}).status_code==422
        assert client.get('/recommend',params={'method':'arbitrary'}).status_code==422
        assert client.get('/recommend',params={'query':'zzzzzzzzzzzzz'}).status_code==422
        assert client.get('/report').json()['test']['target_users']>0
    print('API CHECK PASSED: real artifact retrieval, fallback, input limits and report. Browser layout not tested.')
if __name__=='__main__':main()
