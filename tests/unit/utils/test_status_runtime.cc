#include <gtest/gtest.h>

#include "llm/utils/status.h"

using llm::utils::Status;
using llm::utils::StatusOr;

TEST(StatusTest, DefaultIsOk){
    Status s;
    EXPECT_TRUE(s.ok());
    EXPECT_EQ(s.message(), "");
}

TEST(StatusTest, ErrorHasMessage){
    Status s("fail");
    EXPECT_FALSE(s.ok());
    EXPECT_EQ(s.message(), "fail");
}


TEST(StatusOrTest,okValueBasic){
    StatusOr<int> r(42);
    EXPECT_TRUE(r.ok());
    EXPECT_EQ(r.value(),42);
}

TEST(StatusOrTest, OkAlsoHasOkStatusAndEmptyMessage)
{
    StatusOr<int> r(7);
    EXPECT_TRUE(r.ok());
    EXPECT_TRUE(r.status().ok());
    EXPECT_EQ(r.status().message(), "");
}

TEST(StatusOrTest,ErrorStatusPropagates)
{
    StatusOr<int> r(Status("boom"));
    EXPECT_FALSE(r.ok());
    EXPECT_EQ(r.status().message(), "boom");
}

