#include <boost/test/unit_test.hpp>
#include <dem_generation/Dummy.hpp>

using namespace dem_generation;

BOOST_AUTO_TEST_CASE(it_should_not_crash_when_welcome_is_called)
{
    dem_generation::DummyClass dummy;
    dummy.welcome();
}
