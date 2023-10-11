"""add disabled column

Revision ID: 7ec6c2a6a1c8
Revises: 9e59e0b9d1cf
Create Date: 2020-02-28 16:12:35.548279

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql

# revision identifiers, used by Alembic.
revision = '7ec6c2a6a1c8'
down_revision = '9e59e0b9d1cf'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('flicket_users', sa.Column('disabled', sa.Boolean(), nullable=True))
    # ### end Alembic commands ###

    # update the user column so all values are disabled values are False for
    # user.
    from application import db
    from application.flicket.models.flicket_user import FlicketUser

    users = FlicketUser.query.all()

    if users:
        for user in users:
            if user.disabled is None:
                user.disabled = False

        db.session.commit()


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('flicket_users', 'disabled')
    # ### end Alembic commands ###
