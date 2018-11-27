from __future__ import print_function, division

from nose.tools import assert_equal


def test_drop_www():
    from url_gen.data import drop_www
    assert_equal(drop_www('www.url'), 'url')
    assert_equal(drop_www('www.url.www'), 'url.www')
    assert_equal(drop_www('wwwurl'), 'wwwurl')
    assert_equal(drop_www('url'), 'url')


def test_drop_http():
    from url_gen.data import drop_http
    assert_equal(drop_http('http://url'), 'url')
    assert_equal(drop_http('http://url.www'), 'url.www')
    assert_equal(drop_http('httpurl'), 'httpurl')
    assert_equal(drop_http('url'), 'url')


def test_base_url_only():
    from url_gen.data import base_url_only
    assert_equal(base_url_only('url/page'), 'url')
    assert_equal(base_url_only('url.x/page'), 'url.x')
    assert_equal(base_url_only('url/page/subpage'), 'url')
    assert_equal(base_url_only('http://url/page'), 'http://url')
